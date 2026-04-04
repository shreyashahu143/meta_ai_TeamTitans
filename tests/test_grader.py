"""
tests/test_grader.py — Unit Tests for Scoring Functions
=========================================================

Run: pytest tests/test_grader.py -v

Tests: grade_task_1/2/3, calculate_priority_accuracy,
       calculate_vip_handling_score, calculate_avg_relationship_health,
       calculate_time_efficiency, calculate_value_efficiency
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from models import State, Email, Relationship
from grader import (
    grade_episode,
    grade_task_1,
    grade_task_2,
    grade_task_3,
    calculate_priority_accuracy,
    calculate_vip_handling_score,
    calculate_avg_relationship_health,
    calculate_time_efficiency,
    calculate_value_efficiency,
)


# ---------------------------------------------------------------------------
# FIXTURES — reusable fake data
# ---------------------------------------------------------------------------

def make_email(email_id=0, sender="test@co.com", importance="Normal",
               priority=5, is_followup=False) -> Email:
    return Email(
        email_id=email_id,
        sender=sender,
        sender_domain="internal",
        subject="Test Subject",
        body="Test body content.",
        sender_importance=importance,
        base_priority=priority,
        estimated_response_time=15,
        is_followup=is_followup,
    )


def make_relationship(sender="test@co.com", importance="Normal",
                       health=75.0, is_angry=False) -> Relationship:
    weight_map = {"VIP": 3, "Normal": 2, "Spam": 0}
    return Relationship(
        sender_email=sender,
        health=health,
        importance=importance,
        importance_weight=weight_map[importance],
        degradation_rate=10.0 if importance == "Normal" else 20.0,
        is_angry=is_angry,
    )


def make_state(inbox=None, relationships=None, time_spent=100) -> State:
    inbox = inbox or [make_email()]
    relationships = relationships or {"test@co.com": make_relationship()}
    return State(
        inbox=inbox,
        relationships=relationships,
        current_email_index=len(inbox),
        current_timestep=len(inbox),
        time_budget_remaining=380,
        total_time_spent=time_spent,
        emails_handled=len(inbox),
    )


def make_step(action=1, importance="Normal", priority=5, reward=3.0):
    return {
        "step": 1,
        "observation": {
            "email_id": 0,
            "sender": "test@co.com",
            "subject": "Test",
            "body": "Body",
            "sender_importance": importance,
            "email_length": 100,
            "relationship_score": 75.0,
            "time_budget_remaining": 400,
            "emails_remaining": 5,
        },
        "action": action,
        "reward": reward,
        "info": {},
    }


# ---------------------------------------------------------------------------
# grade_episode — routing
# ---------------------------------------------------------------------------

def test_grade_episode_task1_returns_0_to_100():
    history = [make_step(action=1, importance="VIP", priority=9, reward=5.0)]
    state = make_state(
        inbox=[make_email(importance="VIP", priority=9)],
        relationships={"test@co.com": make_relationship("test@co.com", "VIP", health=90.0)},
    )
    score = grade_episode(history, state, task_id=1)
    assert 0 <= score <= 100


def test_grade_episode_task2_returns_0_to_100():
    history = [make_step(action=1, importance="VIP", priority=8, reward=4.0)]
    state = make_state(
        relationships={"test@co.com": make_relationship("test@co.com", "VIP", health=80.0)}
    )
    score = grade_episode(history, state, task_id=2)
    assert 0 <= score <= 100


def test_grade_episode_task3_returns_0_to_100():
    history = [make_step(action=1, importance="Normal", priority=6, reward=3.0)]
    state = make_state()
    score = grade_episode(history, state, task_id=3)
    assert 0 <= score <= 100


def test_grade_episode_invalid_task_raises():
    with pytest.raises(ValueError):
        grade_episode([], make_state(), task_id=99)


# ---------------------------------------------------------------------------
# calculate_priority_accuracy
# ---------------------------------------------------------------------------

def test_priority_accuracy_perfect_vip_responds():
    """Responding to all VIPs → high accuracy."""
    history = [make_step(action=1, importance="VIP") for _ in range(5)]
    score = calculate_priority_accuracy(history)
    assert score == 1.0


def test_priority_accuracy_perfect_spam_ignores():
    """Ignoring all spam → high accuracy."""
    history = [make_step(action=0, importance="Spam") for _ in range(5)]
    score = calculate_priority_accuracy(history)
    assert score == 1.0


def test_priority_accuracy_wrong_decisions():
    """Ignoring VIPs → lower accuracy."""
    history = [make_step(action=0, importance="VIP") for _ in range(5)]
    score = calculate_priority_accuracy(history)
    assert score < 1.0


def test_priority_accuracy_empty_history():
    score = calculate_priority_accuracy([])
    assert score == 0.0


def test_priority_accuracy_mixed():
    history = [
        make_step(action=1, importance="VIP"),    # correct
        make_step(action=0, importance="Spam"),   # correct
        make_step(action=0, importance="VIP"),    # wrong
        make_step(action=1, importance="Spam"),   # wrong
    ]
    score = calculate_priority_accuracy(history)
    assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# calculate_avg_relationship_health
# ---------------------------------------------------------------------------

def test_avg_relationship_health_all_perfect():
    state = make_state(relationships={
        "vip@co.com": make_relationship("vip@co.com", "VIP", health=100.0),
        "normal@co.com": make_relationship("normal@co.com", "Normal", health=100.0),
    })
    score = calculate_avg_relationship_health(state)
    assert score == 1.0


def test_avg_relationship_health_all_zero():
    state = make_state(relationships={
        "vip@co.com": make_relationship("vip@co.com", "VIP", health=0.0),
    })
    score = calculate_avg_relationship_health(state)
    assert score == 0.0


def test_avg_relationship_health_spam_ignored():
    """Spam relationship health should NOT affect the score."""
    state = make_state(relationships={
        "vip@co.com": make_relationship("vip@co.com", "VIP", health=100.0),
        "spam@junk.com": make_relationship("spam@junk.com", "Spam", health=0.0),
    })
    score = calculate_avg_relationship_health(state)
    assert score == 1.0  # Spam ignored, VIP at 100 → perfect


def test_avg_relationship_health_vip_weighted_more():
    """VIP health should count 3× vs Normal's 2×."""
    state_vip_high = make_state(relationships={
        "vip@co.com": make_relationship("vip@co.com", "VIP", health=100.0),
        "normal@co.com": make_relationship("normal@co.com", "Normal", health=0.0),
    })
    state_normal_high = make_state(relationships={
        "vip@co.com": make_relationship("vip@co.com", "VIP", health=0.0),
        "normal@co.com": make_relationship("normal@co.com", "Normal", health=100.0),
    })
    score_vip_high = calculate_avg_relationship_health(state_vip_high)
    score_normal_high = calculate_avg_relationship_health(state_normal_high)
    assert score_vip_high > score_normal_high


# ---------------------------------------------------------------------------
# calculate_vip_handling_score
# ---------------------------------------------------------------------------

def test_vip_handling_perfect():
    """Respond to all VIPs + healthy relationships → score near 1.0"""
    history = [make_step(action=1, importance="VIP") for _ in range(4)]
    state = make_state(
        inbox=[make_email(importance="VIP")],
        relationships={"test@co.com": make_relationship("test@co.com", "VIP", health=90.0)},
    )
    score = calculate_vip_handling_score(history, state)
    assert score > 0.7


def test_vip_handling_no_vips_in_history():
    """No VIPs → should return 1.0 (not applicable)."""
    history = [make_step(action=1, importance="Normal")]
    state = make_state()
    score = calculate_vip_handling_score(history, state)
    assert score == 1.0


def test_vip_handling_followup_reduces_score():
    """Follow-up emails in inbox should penalize the score."""
    history = [make_step(action=1, importance="VIP")]
    state = make_state(
        inbox=[make_email(importance="VIP", is_followup=True)],
        relationships={"test@co.com": make_relationship("test@co.com", "VIP", health=60.0)},
    )
    score_with_followup = calculate_vip_handling_score(history, state)

    state_no_followup = make_state(
        inbox=[make_email(importance="VIP", is_followup=False)],
        relationships={"test@co.com": make_relationship("test@co.com", "VIP", health=60.0)},
    )
    score_no_followup = calculate_vip_handling_score(history, state_no_followup)

    assert score_with_followup < score_no_followup


# ---------------------------------------------------------------------------
# calculate_time_efficiency
# ---------------------------------------------------------------------------

def test_time_efficiency_zero_time_spent():
    state = make_state(time_spent=0)
    score = calculate_time_efficiency([], state)
    assert score == 0.5  # Neutral default


def test_time_efficiency_high_reward_low_time():
    """High reward per minute = high efficiency."""
    history = [make_step(reward=50.0)]
    state = make_state(time_spent=10)
    score = calculate_time_efficiency(history, state)
    assert score == 1.0  # Capped at 1.0


def test_time_efficiency_low_reward_high_time():
    """Wasting time on low-value emails = low efficiency."""
    history = [make_step(reward=1.0)]
    state = make_state(time_spent=300)
    score = calculate_time_efficiency(history, state)
    assert score < 0.5


# ---------------------------------------------------------------------------
# grade_task_1 formula weights
# ---------------------------------------------------------------------------

def test_task1_relationship_weighted_more_than_value():
    """
    Task 1: relationship_health weight (0.6) > value_efficiency (0.4).
    An agent with perfect relationships but poor value should outscore
    an agent with perfect value but damaged relationships.
    """
    # Perfect relationships, mediocre value
    history_rel = [make_step(action=1, importance="Normal", reward=1.0)]
    state_good_rel = make_state(relationships={
        "test@co.com": make_relationship("test@co.com", "VIP", health=100.0)
    }, time_spent=50)

    # Damaged relationships, high value
    history_val = [make_step(action=1, importance="Normal", reward=20.0)]
    state_bad_rel = make_state(relationships={
        "test@co.com": make_relationship("test@co.com", "VIP", health=10.0)
    }, time_spent=50)

    score_rel = grade_task_1(history_rel, state_good_rel)
    score_val = grade_task_1(history_val, state_bad_rel)

    assert score_rel > score_val
