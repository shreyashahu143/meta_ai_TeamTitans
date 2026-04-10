"""
server/app.py — FastAPI Server Wrapper
=======================================

PURPOSE:
    Thin HTTP wrapper around environment.py.
    No game logic lives here — all logic is in environment.py.

    Endpoints:
        GET  /health  → liveness check
        POST /reset   → start new episode, returns first EmailObservation as JSON
        POST /step    → take action (0 or 1), returns StepResponse as JSON
        GET  /state   → return full State as JSON (for grader/debug)

OWNER: Algorithm Engineer
"""

import os
import sys
 
import uvicorn
 
# Add server/ directory AND project root to sys.path
sys.path.insert(0, os.path.dirname(__file__))                     # server/ folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")) # project root
 
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
 
from environment import EmailTriageEnv
from models import StepResponse
 
app = FastAPI(
    title="Email Triage RL Environment",
    description="OpenEnv email triage environment. POST /reset to start, POST /step to act.",
    version="1.0.0",
)

@app.get("/")
def root():
    return {"status": "ok", "message": "Email Triage RL Environment is running"}
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
env = EmailTriageEnv()
 
 
class StepRequest(BaseModel):
    action: int
 
 
class TaskConfig(BaseModel):
    config: dict = {}
 
 
@app.get("/health")
def health_check():
    return {"status": "ok", "version": "1.0.0"}
 

class ResetRequest(BaseModel):
    task_id: int = 1
    config: dict = {}

@app.post("/reset")
def reset(request: ResetRequest = None):
    global env
    task_id = request.task_id if request else 1
    
    # Load task config from file
    BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
    task_config_path = os.path.join(BASE_DIR, "tasks", f"task_{task_id}_{'easy' if task_id==1 else 'medium' if task_id==2 else 'hard'}.json")
    if os.path.exists(task_config_path):
        with open(task_config_path) as f:
            task_config = json.load(f)
    else:
        task_config = request.config if request and request.config else {}
    
    env = EmailTriageEnv(task_config=task_config if task_config else None)
    observation = env.reset()
    return observation.model_dump()
 
 
@app.post("/step")
def step(request: StepRequest):
    if request.action not in (0, 1):
        raise HTTPException(status_code=400, detail="Action must be 0 or 1")
    response = env.step(request.action)
    return response.model_dump()
 
 
@app.get("/state")
def get_state():
    state = env.state()
    return state.model_dump()
 
 
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)
 
 
if __name__ == "__main__":
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)
 