# AssistX — multi-key AI router bot (Telegram)

AssistX routes every message across **many free AI providers and many API keys**
(OmniRoute-style), makes the answer *smart* whatever model answered, and
remembers your chats.

## What's inside
- **Key pool** — put 50+ keys in `keys.txt`; rotated automatically, bad/limited keys are cooled down or disabled.
- **Model registry** — `app/config_data/models.json` (tiers 1 strong / 2 solid / 3 fast). Edit freely.
- **Smart layer** — language match (Hindi/Hinglish/English), task + difficulty detection, easy → fast model, hard → strongest, bad/looping/empty answers retried on a stronger model, cut-off answers continued, `<think>` removed.
- **Persistent memory** — last 30 messages + rolling summary + permanent facts (`yaad rakhna ki ...`, `/remember`). SQLite locally, Supabase on Heroku.
- **Telegram-safe formatting** — Markdown → HTML, code blocks, long answers split safely.

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env            # set BOT_TOKEN, TELEGRAM_ALLOWED_USER_IDS
cp keys.txt.example keys.txt    # add your keys, one per line:  provider:key
python scripts/check_keys.py --probe   # which keys work, which models they see
python -m app.main
```
Keys never go in git (`keys.txt` is ignored) and never in chat.

## Commands
`/chat /code /content /freelance /summary` · `/memory` · `/remember <text>` · `/forget <n|all>` · `/reset` · `/status`

## Deploy on Heroku (Student Pack credit)
Heroku's disk is wiped on restart, so use **Supabase** for memory.
1. Supabase: run `docs/supabase_schema.sql`; copy project URL + `service_role` key.
2. `heroku create`, then set config vars:
   `BOT_TOKEN`, `TELEGRAM_ALLOWED_USER_IDS`, `MEMORY_BACKEND=supabase`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`,
   and `AI_KEYS="groq:gsk_a,groq:gsk_b,gemini:AIza..."` (keys.txt isn't deployed).
3. `git push heroku main`, then `heroku ps:scale web=0 worker=1` (the bot polls; it needs a **worker** dyno, Eco $5/mo).

## Tests
`pytest` (needs requirements) or, dependency-free: `python -m unittest tests.test_pool_engine tests.test_smart tests.test_agent_router_smart tests.test_memory tests.test_check_keys tests.test_telegram_format`
