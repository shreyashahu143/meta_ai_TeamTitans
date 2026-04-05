"""
grader.py — Scoring Logic for All 3 Tasks
==========================================

WHAT CHANGED:
    - All grade functions now return 0.0 to 1.0 (NOT 0-100)
      This is MANDATORY per OpenEnv spec: "Graders must return within 0.0 to 1.0"
    - print_score_report multiplies by 100 only for human-readable display
    - calculate_priority_accuracy now uses reward signal for Normal emails
      (old version gave free 0.5 credit to every Normal — inflated scores)
    - calculate_value_efficiency now uses the actual reward formula from environment.py
      (old version used priority * 0.5 which was disconnected from reality)
    - calculate_time_efficiency now derives benchmark dynamically from this episode's inbox
      (old version hardcoded BENCHMARK_REWARD_PER_MINUTE = 1.0 which was meaningless)

WHAT DID NOT CHANGE:
    - The 3-task structure and weight formulas
    - calculate_avg_relationship_health logic
    - calculate_vip_handling_score logic
    - print_score_report structure
    - The standalone __main__ test block
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
        episode_history: List of step dicts from inference.py
        final_state:     State object from client.get_state() at episode end
        task_id:         1 (easy), 2 (medium), or 3 (hard)

    Returns:
        float: Score between 0.0 and 1.0  ← SPEC REQUIREMENT
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
    value_score = calculate_value_efficiency(episode_history, final_state)
    rel_score   = calculate_avg_relationship_health(final_state)
    return round(0.4 * value_score + 0.6 * rel_score, 4)


def grade_task_2(episode_history: List[dict], final_state: State) -> float:
    priority_score = calculate_priority_accuracy(episode_history)
    vip_score      = calculate_vip_handling_score(episode_history, final_state)
    return round(0.5 * priority_score + 0.5 * vip_score, 4)


def grade_task_3(episode_history: List[dict], final_state: State) -> float:
    time_score     = calculate_time_efficiency(episode_history, final_state)
    rel_score      = calculate_avg_relationship_health(final_state)
    priority_score = calculate_priority_accuracy(episode_history)
    return round(0.3 * time_score + 0.4 * rel_score + 0.3 * priority_score, 4)


# ---------------------------------------------------------------------------
# METRIC CALCULATORS
# ---------------------------------------------------------------------------

def calculate_value_efficiency(episode_history: List[dict], final_state: State) -> float:
    """
    Did the agent respond to emails worth responding to?
    Uses same reward formula as environment.py for honest comparison.
    Returns float in [0.0, 1.0]
    """
    if not episode_history:
        return 0.0

    total_reward = sum(step["reward"] for step in episode_history)

    max_possible = 0.0
    for email in final_state.inbox:
        normalized_cost  = email.estimated_response_time / 120.0
        estimated_reward = (email.base_priority - 5.0 * normalized_cost) * 0.75
        if estimated_reward > 0:
            max_possible += estimated_reward

    if max_possible <= 0:
        return 0.5

    return min(1.0, max(0.0, total_reward / max_possible))


def calculate_avg_relationship_health(final_state: State) -> float:
    """
    Weighted average relationship health at episode end.
    VIP=weight 3, Normal=weight 2, Spam=ignored.
    Returns float in [0.0, 1.0]
    """
    if not final_state.relationships:
        return 0.5

    weighted_sum = 0.0
    weight_total = 0.0

    for sender, rel in final_state.relationships.items():
        if rel.importance == "Spam":
            continue
        weight        = rel.importance_weight
        weighted_sum  += rel.health * weight
        weight_total  += weight * 100

    if weight_total <= 0:
        return 0.5

    return weighted_sum / weight_total


def calculate_priority_accuracy(episode_history: List[dict]) -> float:
    """
    Did the agent make correct RESPOND/IGNORE decisions?

    VIP + RESPOND  → 1.0 | VIP + IGNORE  → 0.0
    Spam + IGNORE  → 1.0 | Spam + RESPOND → 0.0
    Normal: uses reward signal (reward>0 → 1.0, reward<-1.5 → 0.0, else → 0.5)

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

        if importance == "VIP":
            correct += 1.0 if action == 1 else 0.0
        elif importance == "Spam":
            correct += 1.0 if action == 0 else 0.0
        else:
            reward = step.get("reward", 0)
            if reward > 0:
                correct += 1.0
            elif reward < -1.5:
                correct += 0.0
            else:
                correct += 0.5

    return correct / total


def calculate_vip_handling_score(episode_history: List[dict], final_state: State) -> float:
    """
    VIP-specific handling score.
    50% VIP response rate + 50% avg VIP health - follow-up penalty (max 0.3)
    Returns float in [0.0, 1.0]
    """
    vip_steps = [
        s for s in episode_history
        if s["observation"].get("sender_importance") == "VIP"
    ]

    if not vip_steps:
        return 1.0

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
    Dynamic time efficiency — compares actual reward against perfect agent benchmark.
    Perfect agent picks highest-priority emails first until time budget runs out.
    Returns float in [0.0, 1.0]
    """
    total_time_spent = final_state.total_time_spent
    if total_time_spent <= 0:
        return 0.5

    total_reward = sum(step["reward"] for step in episode_history)
    time_budget  = final_state.time_budget_remaining + total_time_spent

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
# SUMMARY PRINTER
# ---------------------------------------------------------------------------

def print_score_report(episode_history: List[dict], final_state: State, task_id: int):
    score = grade_episode(episode_history, final_state, task_id)

    print(f"\n{'='*50}")
    print(f"  TASK {task_id} SCORE REPORT")
    print(f"{'='*50}")
    print(f"  Final Score (0.0-1.0): {score:.4f}")
    print(f"  Final Score (0-100):   {score * 100:.1f}")

    rel_score      = calculate_avg_relationship_health(final_state)
    priority_score = calculate_priority_accuracy(episode_history)

    print(f"\n  Avg Relationship Health : {rel_score*100:.1f}%")
    print(f"  Priority Accuracy       : {priority_score*100:.1f}%")

    if task_id in [1, 3]:
        val_score = calculate_value_efficiency(episode_history, final_state)
        print(f"  Value Efficiency        : {val_score*100:.1f}%")

    if task_id == 2:
        vip_score = calculate_vip_handling_score(episode_history, final_state)
        print(f"  VIP Handling Score      : {vip_score*100:.1f}%")

    if task_id == 3:
        time_score = calculate_time_efficiency(episode_history, final_state)
        print(f"  Time Efficiency         : {time_score*100:.1f}%")

    print(f"\n  VIP Relationships:")
    for sender, rel in final_state.relationships.items():
        if rel.importance == "VIP":
            status = "✓" if rel.health >= 60 else "✗"
            angry  = " [ANGRY]" if rel.is_angry else ""
            print(f"    {status} {sender}: {rel.health:.0f}/100{angry}")

    print(f"{'='*50}\n")
    return score


# ---------------------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import random
    sys.path.insert(0, ".")
    from client import reset_env, step_env, get_state

    print("Running rule-based episode for grader test...\n")
    obs             = reset_env()
    episode_history = []
    step            = 0

    while step < 100:
        if obs.sender_importance == "VIP":
            action = 1
        elif obs.sender_importance == "Spam":
            action = 0
        else:
            action = random.choice([0, 1])

        result = step_env(action)
        episode_history.append({
            "step":        step + 1,
            "observation": obs.model_dump(),
            "action":      action,
            "reward":      result.reward,
            "info":        result.info,
        })
        step += 1
        if result.done:
            break
        obs = result.observation

    final_state = get_state()

    for task_id in [1, 2, 3]:
        print_score_report(episode_history, final_state, task_id)