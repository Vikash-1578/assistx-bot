"""AgentRouter smart layer: tiering, escalation, continuation, cleaning."""
from __future__ import annotations

import unittest

from app.services.agent_router import AgentRouter, load_policy, load_prompt
from app.services.ai.exceptions import AllProvidersFailed
from app.services.ai.models import AIResponse, ChatMessage, TaskType

GOOD = "Here is a clear and complete answer to your question."
LOOP = "\n".join(["I keep repeating this exact useless sentence."] * 6)


class LegacyEngine:
    """Old-style engine: no tier support, exact AIEngine.generate signature."""

    def __init__(self, texts=None):
        self.texts = list(texts or ["fake reply"])
        self.calls: list[dict] = []

    async def generate(self, messages, *, task, user_id=None,
                       temperature=None, max_tokens=None):
        self.calls.append({"messages": messages, "task": task, "user_id": user_id})
        text = self.texts[min(len(self.calls) - 1, len(self.texts) - 1)]
        return AIResponse(text=text, provider="fake", model="fake-model", total_tokens=42)


class TierEngine:
    supports_tiers = True

    def __init__(self, script):
        # script: list of (text, entry_id, finish_reason) or an Exception
        self.script = list(script)
        self.calls: list[dict] = []

    async def generate(self, messages, *, task, user_id=None, temperature=None,
                       max_tokens=None, prefer="balanced", exclude=None):
        self.calls.append({
            "messages": messages, "task": task, "prefer": prefer,
            "exclude": set(exclude or ()),
        })
        item = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        if isinstance(item, Exception):
            raise item
        text, entry, finish = item
        return AIResponse(text=text, provider="p", model=entry, entry_id=entry,
                          finish_reason=finish)


class LegacyCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    """Mirrors the pre-existing tests/test_routing.py expectations."""

    async def test_chat_keeps_user_text_and_system_first(self):
        engine = LegacyEngine()
        resp = await AgentRouter(engine).run("hello", task=TaskType.CHAT, user_id=1)
        self.assertEqual(resp.text, "fake reply")
        call = engine.calls[0]
        self.assertEqual((call["task"], call["user_id"]), (TaskType.CHAT, 1))
        self.assertEqual(call["messages"][0].role, "system")
        self.assertEqual(call["messages"][-1].content, "hello")

    async def test_explicit_tasks_are_respected(self):
        for task, text in [(TaskType.CODE, "fix this"), (TaskType.CONTENT, "write blog")]:
            engine = LegacyEngine()
            await AgentRouter(engine).run(text, task=task, user_id=2)
            self.assertEqual(engine.calls[0]["task"], task)

    async def test_history_and_extra_system(self):
        engine = LegacyEngine()
        history = [ChatMessage(role="user", content="earlier question"),
                   ChatMessage(role="assistant", content="earlier answer")]
        await AgentRouter(engine).run(
            "follow up", history=history, extra_system="EXTRA_INSTRUCTION", user_id=4)
        msgs = engine.calls[0]["messages"]
        self.assertEqual(len(msgs), 4)
        self.assertEqual([m.content for m in msgs[1:]],
                         ["earlier question", "earlier answer", "follow up"])
        self.assertIn("EXTRA_INSTRUCTION", msgs[0].content)

    async def test_legacy_engine_never_escalates(self):
        engine = LegacyEngine([LOOP])
        resp = await AgentRouter(engine).run("do it", user_id=1)
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(resp.text, LOOP)


class PromptAndTaskTests(unittest.IsolatedAsyncioTestCase):
    async def test_system_prompt_has_policy_and_language_hint(self):
        engine = LegacyEngine()
        await AgentRouter(engine).run("bhai ye kya hai samjhao", user_id=1)
        system = engine.calls[0]["messages"][0].content
        self.assertIn(load_prompt(TaskType.CHAT)[:40], system)
        self.assertIn(load_policy()[:30], system)
        self.assertIn("Hinglish", system)

    async def test_auto_task_only_when_asked(self):
        text = "fix this python error: IndexError in my loop"
        engine = LegacyEngine()
        await AgentRouter(engine).run(text, user_id=1)
        self.assertEqual(engine.calls[0]["task"], TaskType.CHAT)
        engine = LegacyEngine()
        await AgentRouter(engine).run(text, user_id=1, auto_task=True)
        self.assertEqual(engine.calls[0]["task"], TaskType.CODE)
        engine = LegacyEngine()  # explicit non-chat task is never overridden
        await AgentRouter(engine).run(text, task=TaskType.CONTENT, user_id=1, auto_task=True)
        self.assertEqual(engine.calls[0]["task"], TaskType.CONTENT)

    async def test_think_blocks_are_removed(self):
        engine = LegacyEngine(["<think>hmm</think>The answer is 4."])
        resp = await AgentRouter(engine).run("what is 2+2", user_id=1)
        self.assertEqual(resp.text, "The answer is 4.")

    async def test_tier_follows_difficulty(self):
        cases = [("hi", "speed"), ("What is photosynthesis?", "balanced"),
                 ("Design a system architecture for a chat app", "quality")]
        for text, expected in cases:
            engine = TierEngine([(GOOD, "m1", "stop")])
            await AgentRouter(engine).run(text, user_id=1)
            self.assertEqual(engine.calls[0]["prefer"], expected, text)


class EscalationTests(unittest.IsolatedAsyncioTestCase):
    async def test_bad_answer_escalates_to_stronger_model(self):
        engine = TierEngine([(LOOP, "weak", "stop"), (GOOD, "strong", "stop")])
        resp = await AgentRouter(engine).run("explain tcp", user_id=1)
        self.assertEqual((resp.text, resp.entry_id), (GOOD, "strong"))
        self.assertEqual(len(engine.calls), 2)
        second = engine.calls[1]
        self.assertEqual((second["prefer"], second["exclude"]), ("quality", {"weak"}))
        self.assertIn("rejected", second["messages"][0].content)
        self.assertIn("repetition", second["messages"][0].content)

    async def test_gives_up_after_max_escalations_and_returns_best(self):
        engine = TierEngine([(LOOP, "a", "stop"), (LOOP, "b", "stop"), (LOOP, "c", "stop")])
        resp = await AgentRouter(engine, max_escalations=2).run("explain tcp", user_id=1)
        self.assertEqual(len(engine.calls), 3)
        self.assertEqual(resp.text, LOOP)

    async def test_no_stronger_model_left_keeps_first_answer(self):
        engine = TierEngine([(LOOP, "a", "stop"), AllProvidersFailed(["x"])])
        resp = await AgentRouter(engine).run("explain tcp", user_id=1)
        self.assertEqual(resp.text, LOOP)

    async def test_first_call_failure_still_raises(self):
        engine = TierEngine([AllProvidersFailed(["x"])])
        with self.assertRaises(AllProvidersFailed):
            await AgentRouter(engine).run("explain tcp", user_id=1)

    async def test_smart_can_be_disabled(self):
        engine = TierEngine([(LOOP, "a", "stop"), (GOOD, "b", "stop")])
        resp = await AgentRouter(engine, smart_enabled=False).run("explain tcp", user_id=1)
        self.assertEqual((len(engine.calls), resp.text), (1, LOOP))

    async def test_refusal_is_not_retried(self):
        engine = TierEngine([("I can't help with that.", "a", "stop"), (GOOD, "b", "stop")])
        resp = await AgentRouter(engine).run("something bad", user_id=1)
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(resp.entry_id, "a")


class ContinuationTests(unittest.IsolatedAsyncioTestCase):
    async def test_length_cutoff_is_continued(self):
        engine = TierEngine([
            ("Step one is to install the tool and then", "a", "length"),
            ("configure it carefully.", "a", "stop"),
        ])
        resp = await AgentRouter(engine).run("how do I set it up", user_id=1)
        self.assertEqual(resp.text, "Step one is to install the tool and then configure it carefully.")
        cont = engine.calls[1]["messages"]
        self.assertEqual(cont[-2].role, "assistant")
        self.assertIn("cut off", cont[-1].content)

    async def test_unclosed_code_block_is_continued_then_closed(self):
        engine = TierEngine([
            ("Here:\n```python\nx = 1", "a", "stop"),
            ("y = 2", "a", "stop"),
        ])
        resp = await AgentRouter(engine).run("write python", user_id=1)
        self.assertEqual(resp.text, "Here:\n```python\nx = 1\ny = 2\n```")

    async def test_continuation_failure_still_returns_closed_text(self):
        engine = TierEngine([("Code:\n```py\nx = 1", "a", "stop"), AllProvidersFailed(["x"])])
        resp = await AgentRouter(engine).run("write python", user_id=1)
        self.assertTrue(resp.text.endswith("```"))


class SummarizeTests(unittest.IsolatedAsyncioTestCase):
    async def test_summarize_history(self):
        engine = TierEngine([("<think>x</think>User is Vikash, studies BCA.", "s", "stop")])
        out = await AgentRouter(engine).summarize_history(
            "old summary text",
            [ChatMessage(role="user", content="mera naam Vikash hai"),
             ChatMessage(role="assistant", content="Nice to meet you")],
        )
        self.assertEqual(out, "User is Vikash, studies BCA.")
        call = engine.calls[0]
        self.assertEqual((call["task"], call["prefer"]), (TaskType.SUMMARY, "speed"))
        prompt = call["messages"][-1].content
        self.assertIn("old summary text", prompt)
        self.assertIn("user: mera naam Vikash hai", prompt)


if __name__ == "__main__":
    unittest.main()
