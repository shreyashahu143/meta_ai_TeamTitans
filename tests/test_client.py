"""
tests/test_client.py — Integration Tests: Client ↔ Server Roundtrip
=====================================================================

IMPORTANT: These tests require the server to be running.
Start the server first:
    cd server/ && uvicorn app:app --port 8000

Then run:
    pytest tests/test_client.py -v

Tests: reset_env(), step_env(), get_state(), health_check(),
       full episode loop, error handling
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import client
from models import EmailObservation, StepResponse, State


# ---------------------------------------------------------------------------
# SKIP ALL IF SERVER IS NOT RUNNING
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: marks tests that require a running server"
    )


@pytest.fixture(autouse=True)
def require_server():
    """Skip the entire module if the server is not reachable."""
    if not client.health_check():
        pytest.skip(
            "Server is not running. Start it with:\n"
            "  cd server/ && uvicorn app:app --port 8000\n"
            "Then re-run: pytest tests/test_client.py -v"
        )


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------

def test_health_check_returns_true():
    assert client.health_check() is True


# ---------------------------------------------------------------------------
# reset_env
# ---------------------------------------------------------------------------

def test_reset_returns_email_observation():
    obs = client.reset_env()
    assert isinstance(obs, EmailObservation)


def test_reset_observation_has_required_fields():
    obs = client.reset_env()
    assert obs.email_id >= 0
    assert isinstance(obs.sender, str) and "@" in obs.sender
    assert isinstance(obs.subject, str) and len(obs.subject) > 0
    assert obs.sender_importance in ("VIP", "Normal", "Spam")
    assert obs.time_budget_remaining > 0
    assert obs.emails_remaining >= 0
    assert 0.0 <= obs.relationship_score <= 100.0


def test_reset_with_task_config():
    config = {"num_emails": 10, "vip_count": 2, "normal_count": 5, "spam_count": 3, "time_budget": 240}
    obs = client.reset_env(task_config=config)
    assert isinstance(obs, EmailObservation)
    assert obs.time_budget_remaining == 240


def test_reset_is_idempotent():
    """Calling reset twice should give a fresh episode each time."""
    obs1 = client.reset_env()
    obs2 = client.reset_env()
    # Both valid observations (episode restarts cleanly)
    assert isinstance(obs1, EmailObservation)
    assert isinstance(obs2, EmailObservation)


# ---------------------------------------------------------------------------
# step_env
# ---------------------------------------------------------------------------

def test_step_ignore_returns_step_response():
    client.reset_env()
    result = client.step_env(action=0)
    assert isinstance(result, StepResponse)


def test_step_respond_returns_step_response():
    client.reset_env()
    result = client.step_env(action=1)
    assert isinstance(result, StepResponse)


def test_step_response_has_required_fields():
    client.reset_env()
    result = client.step_env(action=1)
    assert isinstance(result.reward, float)
    assert isinstance(result.done, bool)
    assert isinstance(result.observation, EmailObservation)
    assert isinstance(result.info, dict)


def test_step_ignore_reward_non_positive_for_vip():
    """Ignoring a VIP should yield a negative or zero reward."""
    client.reset_env()
    # Take steps until we hit a VIP or exhaust inbox
    for _ in range(20):
        state = client.get_state()
        idx = state.current_email_index
        if idx >= len(state.inbox):
            break
        current = state.inbox[idx]
        if current.sender_importance == "VIP":
            result = client.step_env(action=0)  # IGNORE
            assert result.reward <= 0, f"Expected non-positive reward for ignoring VIP, got {result.reward}"
            return
        client.step_env(action=1)  # Skip past non-VIP

    pytest.skip("No VIP email encountered in this episode")


def test_step_respond_positive_reward_high_priority():
    """Responding to a high-priority email should yield positive reward."""
    client.reset_env()
    for _ in range(20):
        state = client.get_state()
        idx = state.current_email_index
        if idx >= len(state.inbox):
            break
        current = state.inbox[idx]
        if current.base_priority >= 7:
            result = client.step_env(action=1)
            assert result.reward > 0, f"Expected positive reward for high-priority respond, got {result.reward}"
            return
        client.step_env(action=1)

    pytest.skip("No high-priority email encountered in this episode")


def test_step_invalid_action_raises():
    """Sending action=2 should raise an error (server returns 400)."""
    client.reset_env()
    with pytest.raises(RuntimeError):
        client.step_env(action=2)


def test_step_done_flag_when_inbox_empty():
    """Episode should be done after handling all emails."""
    client.reset_env()
    result = None
    for _ in range(50):  # Upper bound to avoid infinite loop
        result = client.step_env(action=1)
        if result.done:
            break
    assert result is not None and result.done is True


# ---------------------------------------------------------------------------
# get_state
# ---------------------------------------------------------------------------

def test_get_state_returns_state():
    client.reset_env()
    state = client.get_state()
    assert isinstance(state, State)


def test_get_state_has_inbox():
    client.reset_env()
    state = client.get_state()
    assert isinstance(state.inbox, list)
    assert len(state.inbox) > 0


def test_get_state_has_relationships():
    client.reset_env()
    state = client.get_state()
    assert isinstance(state.relationships, dict)
    assert len(state.relationships) > 0


def test_get_state_initial_health():
    """All relationships should start at 75 right after reset."""
    client.reset_env()
    state = client.get_state()
    for sender, rel in state.relationships.items():
        assert rel.health == 75.0, f"{sender} health should be 75.0, got {rel.health}"


def test_get_state_is_snapshot_not_live():
    """State captured before a step should differ from state after."""
    client.reset_env()
    state_before = client.get_state()
    client.step_env(action=0)  # IGNORE
    state_after = client.get_state()
    assert state_before.current_email_index != state_after.current_email_index


# ---------------------------------------------------------------------------
# FULL EPISODE LOOP
# ---------------------------------------------------------------------------

def test_full_episode_completes_without_error():
    """Run a complete episode from reset to done using always-respond strategy."""
    obs = client.reset_env()
    assert obs is not None

    total_reward = 0.0
    steps = 0

    while True:
        result = client.step_env(action=1)
        total_reward += result.reward
        steps += 1

        if result.done:
            break

        assert steps < 100, "Episode should not exceed 100 steps"

    assert steps > 0
    final_state = client.get_state()
    assert isinstance(final_state, State)


def test_full_episode_always_ignore():
    """Run a complete episode with always-ignore strategy — should still terminate."""
    client.reset_env()
    steps = 0

    while True:
        result = client.step_env(action=0)
        steps += 1
        if result.done:
            break
        assert steps < 100

    assert steps > 0


def test_episode_reward_lower_when_ignoring_vips():
    """
    Always-respond should outscore always-ignore on total reward
    (because ignoring VIPs generates large negative rewards).
    """
    # Always respond
    client.reset_env()
    total_respond = 0.0
    while True:
        result = client.step_env(action=1)
        total_respond += result.reward
        if result.done:
            break

    # Always ignore
    client.reset_env()
    total_ignore = 0.0
    while True:
        result = client.step_env(action=0)
        total_ignore += result.reward
        if result.done:
            break

    assert total_respond > total_ignore, (
        f"Always-respond ({total_respond:.2f}) should outscore always-ignore ({total_ignore:.2f})"
    )
