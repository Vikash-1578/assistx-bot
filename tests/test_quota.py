"""Quota tracker tests."""
from __future__ import annotations

import pytest

from app.services.ai.quota import QuotaTracker


@pytest.mark.asyncio
async def test_new_provider_zero_usage(tmp_path):
    q = QuotaTracker(str(tmp_path / "q.db"))
    await q.init()
    try:
        snap = await q.snapshot("groq", 1000)
        assert snap["requests"] == 0
        assert snap["remaining"] == 1000
    finally:
        await q.close()


@pytest.mark.asyncio
async def test_record_success(tmp_path):
    q = QuotaTracker(str(tmp_path / "q.db"))
    await q.init()
    try:
        await q.record_success("groq", input_tokens=10, output_tokens=20, total_tokens=30)
        await q.record_success("groq")
        snap = await q.snapshot("groq", 100)
        assert snap["requests"] == 2
        assert snap["input_tokens"] == 10
        assert snap["output_tokens"] == 20
        assert snap["total_tokens"] == 30
        assert snap["remaining"] == 98
    finally:
        await q.close()


@pytest.mark.asyncio
async def test_record_failure(tmp_path):
    q = QuotaTracker(str(tmp_path / "q.db"))
    await q.init()
    try:
        await q.record_failure("groq")
        await q.record_failure("groq")
        assert await q.failures_today("groq") == 2
        # failures do not count as requests
        assert await q.requests_today("groq") == 0
    finally:
        await q.close()


@pytest.mark.asyncio
async def test_quota_exhaustion(tmp_path):
    q = QuotaTracker(str(tmp_path / "q.db"))
    await q.init()
    try:
        for _ in range(5):
            await q.record_success("groq")
        snap = await q.snapshot("groq", 5)
        assert snap["remaining"] == 0
        assert snap["requests"] == 5
    finally:
        await q.close()


@pytest.mark.asyncio
async def test_separate_providers(tmp_path):
    q = QuotaTracker(str(tmp_path / "q.db"))
    await q.init()
    try:
        await q.record_success("groq")
        await q.record_success("gemini")
        await q.record_success("gemini")
        assert await q.requests_today("groq") == 1
        assert await q.requests_today("gemini") == 2
    finally:
        await q.close()
