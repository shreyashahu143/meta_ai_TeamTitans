"""
client.py — HTTP Client (The Waiter)
=====================================

PURPOSE:
    This is what inference.py (the AI agent) imports and calls.
    It handles ALL HTTP communication so inference.py never has to think about URLs or JSON.

    The client inherits from HTTPEnvClient (OpenEnv base class) and implements 3 methods:
        1. _step_payload(action)   → Convert Python action → JSON dict for POST /step
        2. _parse_result(response) → Convert JSON response → EmailObservation Python object
        3. _parse_state(response)  → Convert JSON response → State Python object

    The base class handles: sending HTTP requests, error handling, retry logic.

CONNECTS TO:
    ← inference.py       (imports reset_env, step_env, get_state)
    → server/app.py      (sends requests to /reset, /step, /state endpoints)
    ← models.py          (EmailObservation, State, StepResponse types)

HOW IT WORKS:
    inference.py calls:  client.step_env(action=1`)
    client.py does:      POST http://localhost:7860/step {"action": 1}
    server returns:      {"observation": {...}, "reward": 4.2, "done": false, "info": {...}}
    client.py returns:   StepResponse(observation=EmailObservation(...), reward=4.2, ...)

OWNER: LLM Engineer
"""

import os
import requests
from typing import Optional

from models import EmailObservation, State, StepResponse

# Server URL — override via ENV_SERVER_URL environment variable
SERVER_URL = os.getenv("ENV_SERVER_URL", "http://localhost:7860")


# ---------------------------------------------------------------------------
# PUBLIC FUNCTIONS (what inference.py calls)
# ---------------------------------------------------------------------------

def reset_env(task_config: Optional[dict] = None, task_id: int = 1) -> EmailObservation:
    """
    Start a new episode.

    Args:
        task_config: Optional dict to override episode settings.
                     Example: {"num_emails": 20, "vip_count": 5, "time_budget": 480}
                     If None, uses server defaults (Task 1 settings).

    Returns:
        EmailObservation: The first email to evaluate.

    Example (inference.py):
        obs = reset_env()
        print(f"First email: {obs.subject} from {obs.sender}")
    """
    payload = {"task_id": task_id, "config": task_config or {}}
    response = _post("/reset", payload)
    return _parse_observation(response)


def step_env(action: int) -> StepResponse:
    """
    Take one action and get the result.

    Args:
        action: 0 = IGNORE, 1 = RESPOND

    Returns:
        StepResponse with:
            .observation  → next EmailObservation (what to see next)
            .reward       → float reward for this action
            .done         → True if episode is over
            .info         → dict with debug info (time_cost, relationship_delta, etc.)

    Example (inference.py):
        result = step_env(action=1)
        if result.done:
            print(f"Episode ended! Final reward: {result.reward}")
        else:
            next_email = result.observation
    """
    response = _post("/step", {"action": action})
    return _parse_step_response(response)


def get_state() -> State:
    """
    Get the full internal state (God-mode view).

    Use this for:
        - Grader: to check relationship health at end of episode
        - Debugging: to verify what the environment sees vs what agent sees

    Returns:
        State: Full environment state including all emails and relationships.

    Example (grader.py):
        final_state = get_state()
        avg_health = sum(r.health for r in final_state.relationships.values()) / len(final_state.relationships)
    """
    response = _get("/state")
    return _parse_state(response)


def health_check() -> bool:
    """
    Check if the server is running.
    Call this before starting inference to avoid confusing errors.

    Returns:
        True if server is up, False otherwise.
    """
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.ConnectionError:
        return False


# ---------------------------------------------------------------------------
# PRIVATE HELPERS (the 3 translation methods from OpenEnv spec)
# ---------------------------------------------------------------------------

def _step_payload(action: int) -> dict:
    """
    METHOD 1: Action → JSON
    Convert Python action integer into the JSON payload for POST /step.
    """
    return {"action": action}


def _parse_result(response_json: dict) -> EmailObservation:
    """
    METHOD 2: JSON → Observation
    Convert the JSON response from POST /step into an EmailObservation Python object.
    """
    return _parse_observation(response_json.get("observation", response_json))


def _parse_state(response_json: dict) -> State:
    """
    METHOD 3: JSON → State
    Convert the JSON response from GET /state into a State Python object.
    """
    # Rebuild relationship objects
    relationships = {}
    for sender, rel_data in response_json.get("relationships", {}).items():
        from models import Relationship
        relationships[sender] = Relationship(**rel_data)

    # Rebuild email objects
    from models import Email
    inbox = [Email(**e) for e in response_json.get("inbox", [])]

    return State(
        inbox=inbox,
        current_email_index=response_json.get("current_email_index", 0),
        relationships=relationships,
        current_timestep=response_json.get("current_timestep", 0),
        time_budget_remaining=response_json.get("time_budget_remaining", 480),
        total_time_spent=response_json.get("total_time_spent", 0),
        emails_handled=response_json.get("emails_handled", 0),
    )


def _parse_observation(data: dict) -> EmailObservation:
    """Helper: dict → EmailObservation"""
    return EmailObservation(
        email_id=data["email_id"],
        sender=data["sender"],
        subject=data["subject"],
        body=data["body"],
        sender_importance=data["sender_importance"],
        email_length=data["email_length"],
        relationship_score=data["relationship_score"],
        time_budget_remaining=data["time_budget_remaining"],
        emails_remaining=data["emails_remaining"],
    )


def _parse_step_response(data: dict) -> StepResponse:
    """Helper: dict → StepResponse"""
    return StepResponse(
        observation=_parse_observation(data["observation"]),
        reward=data["reward"],
        done=data["done"],
        info=data.get("info", {}),
    )


def _post(endpoint: str, payload: dict) -> dict:
    """Send a POST request and return parsed JSON."""
    url = f"{SERVER_URL}{endpoint}"
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.ConnectionError:
        raise ConnectionError(
            f"Cannot connect to server at {SERVER_URL}. "
            "Is the server running? Start it with: uvicorn server.app:app --port 8000"
        )
    except requests.HTTPError as e:
        raise RuntimeError(f"Server returned error: {e.response.status_code} — {e.response.text}")


def _get(endpoint: str) -> dict:
    """Send a GET request and return parsed JSON."""
    url = f"{SERVER_URL}{endpoint}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.ConnectionError:
        raise ConnectionError(f"Cannot connect to server at {SERVER_URL}.")
