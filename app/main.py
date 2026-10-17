import logging

from fastapi import FastAPI

from app.api.routes import datasets, health, playground, runs, sessions

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="crusoe-hack",
    description="LLM improvement loop: Crusoe inference, Modal sandboxes, Neon Postgres + storage",
    version="0.1.0",
)

app.include_router(health.router)
app.include_router(datasets.router)
app.include_router(runs.router)
app.include_router(playground.router)
app.include_router(sessions.router)
