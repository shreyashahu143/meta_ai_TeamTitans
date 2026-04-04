"""
server/app.py — FastAPI Server Wrapper
=======================================

PURPOSE:
    This makes the environment accessible over HTTP.
    It's a thin wrapper — NO game logic lives here. All logic is in environment.py.

    Exposes 4 endpoints:
        GET  /health    → Check server is alive
        POST /reset     → Start new episode, returns first EmailObservation as JSON
        POST /step      → Take action (0 or 1), returns StepResponse as JSON
        GET  /state     → Return full internal State as JSON (for grader/debug)

HOW IT WORKS:
    Client sends HTTP request → app.py deserializes JSON → calls environment.py
    → environment.py returns Python objects → app.py serializes to JSON → client receives

CONNECTS TO:
    ← server/environment.py  (instantiates EmailTriageEnv, calls reset/step/state)
    → client.py              (client sends requests to these endpoints)

RUN COMMAND:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

OWNER: Algorithm Engineer
"""

import json
import os
import sys

import uvicorn

# Make sure models.py (one level up) is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from server.environment import EmailTriageEnv
from models import StepResponse

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Email Triage RL Environment",
    description="OpenEnv-compatible email triage environment. POST /reset to start, POST /step to act.",
    version="1.0.0",
)

# Allow requests from any origin (needed for Hugging Face Spaces demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single environment instance (stateful — one episode at a time)
env = EmailTriageEnv()


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class StepRequest(BaseModel):
    """
    Input to POST /step.
    action: 0 = IGNORE, 1 = RESPOND
    """
    action: int  # Must be 0 or 1


class TaskConfig(BaseModel):
    """
    Optional input to POST /reset.
    If not provided, uses Task 1 defaults.
    Pass task config dict from tasks/task_1_easy.json etc.
    """
    config: dict = {}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    """
    Simple liveness check.
    Call this after starting Docker to verify the server is up before running inference.py.
    """
    return {"status": "ok", "version": "1.0.0"}


@app.post("/reset")
def reset(task_config: TaskConfig = None):
    """
    Start a new episode.

    Optionally pass a task config (from task_1_easy.json etc.) to set:
        - num_emails, vip_count, normal_count, spam_count, time_budget

    Returns: EmailObservation (the first email to evaluate)

    Example:
        curl -X POST http://localhost:8000/reset
        curl -X POST http://localhost:8000/reset -d '{"config": {"num_emails": 20}}' -H "Content-Type: application/json"
    """
    global env
    if task_config and task_config.config:
        env = EmailTriageEnv(task_config=task_config.config)
    else:
        env = EmailTriageEnv()

    observation = env.reset()
    return observation.dict()


@app.post("/step")
def step(request: StepRequest):
    """
    Take one action in the environment.

    Args:
        action: 0 (IGNORE) or 1 (RESPOND)

    Returns: StepResponse with keys:
        - observation: next EmailObservation
        - reward: float
        - done: bool
        - info: dict with debug info

    Example:
        curl -X POST http://localhost:8000/step \\
             -H "Content-Type: application/json" \\
             -d '{"action": 1}'
    """
    if request.action not in (0, 1):
        raise HTTPException(status_code=400, detail="Action must be 0 (ignore) or 1 (respond)")

    response = env.step(request.action)
    return response.dict()


@app.get("/state")
def get_state():
    """
    Return the full internal state (God-mode view).

    Used by:
        - grader.py: to score relationship management
        - Debugging: to verify environment logic

    NOTE: This is a DEEP COPY of internal state. Safe to call at any time.

    Example:
        curl http://localhost:8000/state
    """
    state = env.state()
    return state.dict()

def main():
    # Uvicorn ko programmatically run karne ka tarika
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
