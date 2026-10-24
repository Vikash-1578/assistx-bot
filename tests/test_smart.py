"""Smart-answer helpers: language, classification, cleaning, quality gate."""
from __future__ import annotations

import unittest

from app.services.ai.models import TaskType
from app.services.smart import (
    classify,
    clean_output,
    detect_language,
    has_unclosed_fence,
    merge_continuation,
    quality_check,
)


class LanguageTests(unittest.TestCase):
    def test_detection(self):
        cases = {
            "mujhe python seekhna hai, kaise start karu": "hinglish",
            "bhai ye error kya hai": "hinglish",
            "kaise ho": "hinglish",
            "मुझे पायथन सीखना है": "hindi",
            "How do I learn Python quickly?": "english",
            "Explain how a hash map works": "english",
            "你好，请帮我写代码": "other",
            "": "english",
        }
        for text, expected in cases.items():
            self.assertEqual(detect_language(text), expected, text)


class ClassifyTests(unittest.TestCase):
    def task(self, text):
        return classify(text).task

    def test_tasks(self):
        self.assertEqual(self.task("fix this python error: IndexError"), TaskType.CODE)
        self.assertEqual(self.task("```js\nconsole.log(1)\n``` why fails"), TaskType.CODE)
        self.assertEqual(self.task("write a youtube script on space facts"), TaskType.CONTENT)
        self.assertEqual(self.task("proposal likho upwork client ke liye"), TaskType.FREELANCE)
        self.assertEqual(self.task("summarize this article for me"), TaskType.SUMMARY)
        self.assertEqual(self.task("solve 12*7 + 5 step by step"), TaskType.REASONING)
        self.assertEqual(self.task("what is the capital of France"), TaskType.CHAT)
        self.assertEqual(self.task("mere class ka timing kya hai"), TaskType.CHAT)
        self.assertEqual(self.task("hello"), TaskType.CHAT)

    def test_vision_when_image(self):
        self.assertEqual(classify("what is this", has_image=True).task, TaskType.VISION)

    def test_difficulty(self):
        self.assertEqual(classify("hi").difficulty, "easy")
        self.assertEqual(classify("thanks").difficulty, "easy")
        self.assertEqual(classify("What is photosynthesis?").difficulty, "normal")
        self.assertEqual(classify("2+2?").difficulty, "normal")  # never "easy"
        self.assertEqual(
            classify("Design a system architecture for a chat app").difficulty, "hard")
        self.assertEqual(
            classify("1. what is X\n2. what is Y\n3. what is Z").difficulty, "hard")
        self.assertEqual(classify("a " * 400).difficulty, "hard")

    def test_language_is_reported(self):
        self.assertEqual(classify("kya hai ye bhai").language, "hinglish")


class CleanOutputTests(unittest.TestCase):
    def test_strips_think_blocks(self):
        raw = "<think>let me reason\nhmm</think>\n\nThe answer is 4."
        self.assertEqual(clean_output(raw), "The answer is 4.")
        self.assertEqual(clean_output("A<THINK>x</THINK>B"), "AB")

    def test_unclosed_think(self):
        self.assertEqual(clean_output("Intro text <think>never closed"), "Intro text")
        self.assertEqual(clean_output("<think>only this"), "only this")

    def test_whitespace_and_control_chars(self):
        self.assertEqual(clean_output("a\x00b\n\n\n\nc  "), "ab\n\nc")
        self.assertEqual(clean_output(""), "")


class ContinuationTests(unittest.TestCase):
    def test_fence_detection(self):
        self.assertTrue(has_unclosed_fence("text\n```python\nx = 1"))
        self.assertFalse(has_unclosed_fence("```py\nx\n```"))
        self.assertFalse(has_unclosed_fence("no code"))

    def test_merge(self):
        self.assertEqual(merge_continuation("Hello wor", "ld again"), "Hello wor ld again")
        self.assertEqual(
            merge_continuation("```py\nx = 1", "y = 2\n```"), "```py\nx = 1\ny = 2\n```")
        self.assertEqual(merge_continuation("Done.", ""), "Done.")
        self.assertEqual(
            merge_continuation("This is the end of the long sentence here",
                               "the end of the long sentence here and more"),
            "This is the end of the long sentence here and more")


class QualityGateTests(unittest.TestCase):
    def check(self, user, ans, **kw):
        return quality_check(user, ans, **kw)

    def test_good_answers_pass(self):
        self.assertTrue(self.check("hi", "Hello! How can I help?").ok)
        self.assertTrue(self.check("2+2?", "4").ok)
        self.assertTrue(self.check(
            "kaise start karu python", "Dekho, pehle Python install karo.",
            language="hinglish").ok)

    def test_empty_and_short(self):
        self.assertEqual(self.check("x", "  ").issues, ["empty"])
        long_q = "Please explain in detail how the TCP three-way handshake works " * 2
        self.assertIn("too_short", self.check(long_q, "Ok.").issues)

    def test_echo(self):
        q = "Tell me about the history of the Roman Empire"
        self.assertIn("echo", self.check(q, q + "!").issues)

    def test_prompt_leak(self):
        r = self.check("hello there", 'You are "Universal AI" — a warm assistant ...')
        self.assertIn("prompt_leak", r.issues)

    def test_repetition_loop(self):
        looped = "\n".join(["I am sorry I cannot continue this task."] * 6)
        self.assertIn("repetition", self.check("do it", looped).issues)
        words = " ".join(["the quick brown fox jumps over the lazy dog"] * 20)
        self.assertIn("repetition", self.check("do it", words).issues)

    def test_repeated_short_code_lines_are_fine(self):
        code = "```c\n" + "}\n" * 8 + "```"
        self.assertTrue(self.check("write c code", code, task=TaskType.CODE).ok)

    def test_wrong_language(self):
        hindi_q = "मुझे पायथन के बारे में बताओ"
        english_ans = "Python is a popular programming language used for many things."
        self.assertIn("wrong_language",
                      self.check(hindi_q, english_ans, language="hindi").issues)
        chinese_ans = "Python 是一种非常流行的编程语言，广泛用于数据分析和人工智能开发。" * 2
        self.assertIn("wrong_language",
                      self.check("tell me about python", chinese_ans, language="english").issues)
        devanagari_ans = "पायथन एक लोकप्रिय प्रोग्रामिंग भाषा है जो बहुत उपयोगी है और आसान है।"
        self.assertIn("wrong_language",
                      self.check("python kya hai bhai", devanagari_ans, language="hinglish").issues)

    def test_refusals_are_not_treated_as_quality_failures(self):
        refusal = "I can't help with that request."
        self.assertTrue(self.check("something I should not get", refusal).ok)


if __name__ == "__main__":
    unittest.main()
