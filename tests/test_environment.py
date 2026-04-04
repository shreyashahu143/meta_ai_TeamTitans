"""
tests/test_environment.py — Unit Tests for the RL Environment
==============================================================

Run: pytest tests/test_environment.py -v

Tests: reset(), step(), reward math, done conditions, relationship decay
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

import pytest
from server.environment import EmailTriageEnv


@pytest.fixture
def env():
    """Fresh environment for each test."""
    return EmailTriageEnv()


def test_reset_returns_observation(env):
    obs = env.reset()
    assert obs.email_id >= 0
    assert obs.sender != ""
    assert obs.time_budget_remaining == 480


def test_reset_populates_inbox(env):
    env.reset()
    state = env.state()
    assert len(state.inbox) == 20  # Task 1 default: 20 emails


def test_reset_initializes_relationships(env):
    env.reset()
    state = env.state()
    assert len(state.relationships) > 0
    for rel in state.relationships.values():
        assert rel.health == 75.0  # All start at 75


def test_step_respond_decreases_time(env):
    env.reset()
    state_before = env.state()
    env.step(1)  # RESPOND
    state_after = env.state()
    assert state_after.time_budget_remaining < state_before.time_budget_remaining


def test_step_ignore_preserves_time(env):
    env.reset()
    state_before = env.state()
    env.step(0)  # IGNORE
    state_after = env.state()
    assert state_after.time_budget_remaining == state_before.time_budget_remaining


def test_step_respond_increases_relationship_health(env):
    env.reset()
    state_before = env.state()
    obs = env.state()
    # Find sender of first email
    first_email = state_before.inbox[0]
    sender = first_email.sender

    if first_email.sender_importance != "Spam":
        health_before = state_before.relationships[sender].health
        env.step(1)  # RESPOND
        state_after = env.state()
        health_after = state_after.relationships[sender].health
        assert health_after >= health_before


def test_step_ignore_vip_sets_angry(env):
    env.reset()
    state = env.state()

    # Find a VIP email
    vip_email = next((e for e in state.inbox if e.sender_importance == "VIP"), None)
    if vip_email is None:
        pytest.skip("No VIP email in this episode")

    # Advance to VIP email
    for i in range(state.inbox.index(vip_email)):
        env.step(1)  # Skip past earlier emails

    # Ignore the VIP
    env.step(0)
    state_after = env.state()
    assert state_after.relationships[vip_email.sender].is_angry == True


def test_done_when_inbox_empty(env):
    env.reset()
    state = env.state()
    num_emails = len(state.inbox)

    result = None
    for _ in range(num_emails):
        result = env.step(1)

    assert result.done == True


def test_reward_positive_for_high_priority_respond(env):
    env.reset()
    state = env.state()
    # Find high-priority email
    high_priority = next((e for e in state.inbox if e.base_priority >= 7), None)
    if high_priority is None:
        pytest.skip("No high priority email")

    for i in range(state.inbox.index(high_priority)):
        env.step(1)

    result = env.step(1)  # Respond to high priority
    assert result.reward > 0


def test_state_is_deep_copy(env):
    env.reset()
    state1 = env.state()
    env.step(0)
    state2 = env.state()
    # They should differ — step 0 advanced the index
    assert state1.current_email_index != state2.current_email_index
