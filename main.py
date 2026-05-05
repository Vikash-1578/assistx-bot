# main.py

import os
from fastapi import FastAPI, Request
from telegram import Update
from telegram.ext import ApplicationBuilder
from dotenv import load_dotenv
from bot import setup_handlers

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

app = FastAPI()

telegram_app = ApplicationBuilder().token(BOT_TOKEN).build()

setup_handlers(telegram_app)


@app.on_event("startup")
async def startup():
    await telegram_app.initialize()
    await telegram_app.start()


@app.get("/")
async def root():
    return {"status": "Ultra Voice AI Bot Running"}


@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, telegram_app.bot)
    await telegram_app.process_update(update)
    return {"ok": True}