"""
grader.py — Scoring Logic for All 3 Tasks
==========================================

SPEC REQUIREMENT: All grade functions return float in [0.0, 1.0]
The evaluators expect scores in this range. Multiply by 100 only for human display.

TASK FORMULAS:
    Task 1 (Easy):   0.4 × value_efficiency   + 0.6 × relationship_health
    Task 2 (Medium): 0.5 × priority_accuracy   + 0.5 × vip_handling_score
    Task 3 (Hard):   0.3 × time_efficiency     + 0.4 × relationship_health + 0.3 × priority_accuracy

WHAT CHANGED FROM ORIGINAL:
    - All grade functions now return 0.0–1.0 (NOT 0–100) — mandatory per OpenEnv spec
    - calculate_priority_accuracy uses reward signal for Normal emails (more accurate)
    - calculate_value_efficiency uses same reward formula as environment.py
    - calculate_time_efficiency derives perfect-agent benchmark dynamically per episode
    - print_score_report multiplies by 100 only for human-readable display

OWNER: LLM Engineer
"""

from typing import List
from models import State


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def grade_episode(episode_history: List[dict], final_state: State, task_id: int) -> float:
    """
    Grade a completed episode.

    Args:
        episode_history: List of step dicts — each: {step, observation, action, reward, info}
        final_state:     State object from client.get_state() at episode end
        task_id:         1 (easy), 2 (medium), or 3 (hard)

    Returns:
        float in [0.0, 1.0]  ← SPEC REQUIREMENT
    """
    if task_id == 1:
        return grade_task_1(episode_history, final_state)
    elif task_id == 2:
        return grade_task_2(episode_history, final_state)
    elif task_id == 3:
        return grade_task_3(episode_history, final_state)
    else:
        raise ValueError(f"Unknown task_id: {task_id}. Must be 1, 2, or 3.")


# ---------------------------------------------------------------------------
# TASK-SPECIFIC GRADERS
# ---------------------------------------------------------------------------

def grade_task_1(episode_history: List[dict], final_state: State) -> float:
    """
    Task 1: Basic Prioritization
    Score = 0.4 × value_efficiency + 0.6 × relationship_health
    Returns float in [0.0, 1.0]
    """
    value_score = calculate_value_efficiency(episode_history, final_state)
    rel_score   = calculate_avg_relationship_health(final_state)
    return round(0.4 * value_score + 0.6 * rel_score, 4)


def grade_task_2(episode_history: List[dict], final_state: State) -> float:
    """
    Task 2: VIP Tracking
    Score = 0.5 × priority_accuracy + 0.5 × vip_handling_score
    Returns float in [0.0, 1.0]
    """
    priority_score = calculate_priority_accuracy(episode_history)
    vip_score      = calculate_vip_handling_score(episode_history, final_state)
    return round(0.5 * priority_score + 0.5 * vip_score, 4)


def grade_task_3(episode_history: List[dict], final_state: State) -> float:
    """
    Task 3: Full Relationship Management
    Score = 0.3 × time_efficiency + 0.4 × relationship_health + 0.3 × priority_accuracy
    Returns float in [0.0, 1.0]
    """
    time_score     = calculate_time_efficiency(episode_history, final_state)
    rel_score      = calculate_avg_relationship_health(final_state)
    priority_score = calculate_priority_accuracy(episode_history)
    return round(0.3 * time_score + 0.4 * rel_score + 0.3 * priority_score, 4)


# ---------------------------------------------------------------------------
# METRIC CALCULATORS — all return float in [0.0, 1.0]
# ---------------------------------------------------------------------------

def calculate_value_efficiency(episode_history: List[dict], final_state: State) -> float:
    """
    Did the agent respond to emails worth responding to?

    Uses same reward formula as environment.py for honest comparison:
        estimated_reward = (base_priority - 5.0 × normalized_cost) × 0.75
    Only counts emails where estimated_reward > 0 (i.e., worth responding to).

    Returns float in [0.0, 1.0]
    """
    if not episode_history:
        return 0.0

    total_reward = sum(step["reward"] for step in episode_history)

    # Compute what a perfect agent would have earned
    max_possible = 0.0
    for email in final_state.inbox:
        normalized_cost  = email.estimated_response_time / 120.0
        estimated_reward = (email.base_priority - 5.0 * normalized_cost) * 0.75
        if estimated_reward > 0:
            max_possible += estimated_reward

    if max_possible <= 0:
        return 0.5  # Neutral if nothing worth responding to

    return min(1.0, max(0.0, total_reward / max_possible))


def calculate_avg_relationship_health(final_state: State) -> float:
    """
    Weighted average relationship health at episode end.
    VIP weight = 3, Normal weight = 2, Spam = excluded.

    Returns float in [0.0, 1.0]
    """
    if not final_state.relationships:
        return 0.5

    weighted_sum = 0.0
    weight_total = 0.0

    for sender, rel in final_state.relationships.items():
        if rel.importance == "Spam":
            continue
        weight       = rel.importance_weight   # VIP=3, Normal=2
        weighted_sum += rel.health * weight
        weight_total += weight * 100           # Max possible contribution

    if weight_total <= 0:
        return 0.5

    return weighted_sum / weight_total


def calculate_priority_accuracy(episode_history: List[dict]) -> float:
    """
    Did the agent make correct RESPOND/IGNORE decisions?

    Scoring:
        VIP   + RESPOND → 1.0  |  VIP   + IGNORE  → 0.0
        Spam  + IGNORE  → 1.0  |  Spam  + RESPOND → 0.0
        Normal: uses reward signal
            reward > 0    → 1.0 (good decision)
            reward < -1.5 → 0.0 (bad decision)
            otherwise     → 0.5 (partial credit)

    Returns float in [0.0, 1.0]
    """
    if not episode_history:
        return 0.0

    correct = 0.0
    total   = len(episode_history)

    for step in episode_history:
        obs        = step["observation"]
        action     = step["action"]
        importance = obs.get("sender_importance", "Normal")
        reward     = step.get("reward", 0)

        if importance == "VIP":
            correct += 1.0 if action == 1 else 0.0
        elif importance == "Spam":
            correct += 1.0 if action == 0 else 0.0
        else:  # Normal — use reward signal
            if reward > 0:
                correct += 1.0
            elif reward < -1.5:
                correct += 0.0
            else:
                correct += 0.5

    return correct / total


def calculate_vip_handling_score(episode_history: List[dict], final_state: State) -> float:
    """
    VIP-specific handling:
        50% × VIP response rate
      + 50% × avg VIP relationship health at end
      - follow-up penalty (max 0.3, -0.05 per follow-up email in inbox)

    Returns float in [0.0, 1.0]
    """
    vip_steps = [
        s for s in episode_history
        if s["observation"].get("sender_importance") == "VIP"
    ]

    if not vip_steps:
        return 1.0  # No VIPs → not applicable, full score

    vip_responds      = sum(1 for s in vip_steps if s["action"] == 1)
    vip_response_rate = vip_responds / len(vip_steps)

    vip_health_scores = [
        rel.health / 100
        for rel in final_state.relationships.values()
        if rel.importance == "VIP"
    ]
    avg_vip_health = (
        sum(vip_health_scores) / len(vip_health_scores)
        if vip_health_scores else 0.5
    )

    followup_count   = sum(1 for e in final_state.inbox if e.is_followup)
    followup_penalty = min(0.3, followup_count * 0.05)

    return min(1.0, max(0.0,
        0.5 * vip_response_rate + 0.5 * avg_vip_health - followup_penalty
    ))


def calculate_time_efficiency(episode_history: List[dict], final_state: State) -> float:
    """
    Dynamic time efficiency.
    Compares actual reward to what a perfect greedy agent would have earned
    (picks highest-priority emails first until time budget runs out).

    Returns float in [0.0, 1.0]
    """
    total_time_spent = final_state.total_time_spent
    if total_time_spent <= 0:
        return 0.5  # No time spent — neutral

    total_reward = sum(step["reward"] for step in episode_history)
    time_budget  = final_state.time_budget_remaining + total_time_spent

    # Perfect agent benchmark: pick best emails greedily
    perfect_reward  = 0.0
    cumulative_time = 0

    for email in sorted(final_state.inbox, key=lambda e: -e.base_priority):
        normalized_cost  = email.estimated_response_time / 120.0
        estimated_reward = (email.base_priority - 5.0 * normalized_cost) * 0.75
        if estimated_reward > 0:
            cumulative_time += email.estimated_response_time
            if cumulative_time > time_budget:
                break
            perfect_reward += estimated_reward

    if perfect_reward <= 0:
        return 0.5

    return min(1.0, max(0.0, total_reward / perfect_reward))


# ---------------------------------------------------------------------------
# SUMMARY PRINTER (human-readable — multiplies by 100 for display only)
# ---------------------------------------------------------------------------

def print_score_report(episode_history: List[dict], final_state: State, task_id: int):
    """Print a human-readable score report. Score displayed as 0-100 for readability."""
    score = grade_episode(episode_history, final_state, task_id)

    print(f"\n{'='*55}")
    print(f"  TASK {task_id} SCORE REPORT")
    print(f"{'='*55}")
    print(f"  Final Score (0.0–1.0): {score:.4f}")
    print(f"  Final Score (0–100):   {score * 100:.1f}")

    rel_score      = calculate_avg_relationship_health(final_state)
    priority_score = calculate_priority_accuracy(episode_history)

    print(f"\n  Avg Relationship Health : {rel_score * 100:.1f}%")
    print(f"  Priority Accuracy       : {priority_score * 100:.1f}%")

    if task_id in (1, 3):
        val_score = calculate_value_efficiency(episode_history, final_state)
        print(f"  Value Efficiency        : {val_score * 100:.1f}%")

    if task_id == 2:
        vip_score = calculate_vip_handling_score(episode_history, final_state)
        print(f"  VIP Handling Score      : {vip_score * 100:.1f}%")

    if task_id == 3:
        time_score = calculate_time_efficiency(episode_history, final_state)
        print(f"  Time Efficiency         : {time_score * 100:.1f}%")

    print(f"\n  VIP Relationships:")
    for sender, rel in final_state.relationships.items():
        if rel.importance == "VIP":
            status = "✓" if rel.health >= 60 else "✗"
            angry  = " [ANGRY]" if rel.is_angry else ""
            print(f"    {status} {sender}: {rel.health:.0f}/100{angry}")

    print(f"{'='*55}\n")
    return score