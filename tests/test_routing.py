"""AgentRouter — prompt loading and message assembly tests."""
from __future__ import annotations

import pytest

from app.services.agent_router import AgentRouter, load_prompt
from app.services.ai.models import AIResponse, ChatMessage, TaskType


class _FakeEngine:
    def __init__(self):
        self.calls: list[dict] = []

    async def generate(self, messages, *, task, user_id=None,
                       temperature=None, max_tokens=None):
        self.calls.append({
            "messages": messages,
            "task": task,
            "user_id": user_id,
        })
        return AIResponse(
            text="fake reply",
            provider="fake",
            model="fake-model",
            total_tokens=42,
        )


def test_load_prompt_all_tasks():
    for task in TaskType:
        prompt = load_prompt(task)
        assert isinstance(prompt, str)
        assert len(prompt) > 10


@pytest.mark.asyncio
async def test_router_chat():
    engine = _FakeEngine()
    router = AgentRouter(engine)  # type: ignore[arg-type]
    resp = await router.run("hello", task=TaskType.CHAT, user_id=1)
    assert resp.text == "fake reply"
    call = engine.calls[0]
    assert call["task"] == TaskType.CHAT
    assert call["user_id"] == 1
    msgs = call["messages"]
    assert msgs[0].role == "system"
    assert msgs[-1].role == "user"
    assert msgs[-1].content == "hello"


@pytest.mark.asyncio
async def test_router_code():
    engine = _FakeEngine()
    router = AgentRouter(engine)  # type: ignore[arg-type]
    await router.run("fix this", task=TaskType.CODE, user_id=2)
    assert engine.calls[0]["task"] == TaskType.CODE


@pytest.mark.asyncio
async def test_router_content():
    engine = _FakeEngine()
    router = AgentRouter(engine)  # type: ignore[arg-type]
    await router.run("write blog", task=TaskType.CONTENT, user_id=3)
    assert engine.calls[0]["task"] == TaskType.CONTENT


@pytest.mark.asyncio
async def test_router_with_history():
    engine = _FakeEngine()
    router = AgentRouter(engine)  # type: ignore[arg-type]
    history = [
        ChatMessage(role="user", content="earlier question"),
        ChatMessage(role="assistant", content="earlier answer"),
    ]
    await router.run("follow up", task=TaskType.CHAT, history=history, user_id=4)
    msgs = engine.calls[0]["messages"]
    # system + 2 history + 1 current = 4
    assert len(msgs) == 4
    assert msgs[1].content == "earlier question"
    assert msgs[2].content == "earlier answer"
    assert msgs[3].content == "follow up"


@pytest.mark.asyncio
async def test_router_extra_system():
    engine = _FakeEngine()
    router = AgentRouter(engine)  # type: ignore[arg-type]
    await router.run(
        "hello",
        task=TaskType.CHAT,
        extra_system="EXTRA_INSTRUCTION",
        user_id=5,
    )
    system = engine.calls[0]["messages"][0]
    assert "EXTRA_INSTRUCTION" in system.content
