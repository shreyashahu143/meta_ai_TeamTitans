"""
grader.py — Scoring Logic for All 3 Tasks
==========================================

PURPOSE:
    Takes an episode's history (all states + actions) and produces a score 0-100.
    Each task has its own scoring formula focusing on different skills.

    Task 1 (Easy):   0.4 × value_efficiency   + 0.6 × relationship_health
    Task 2 (Medium): 0.5 × priority_accuracy   + 0.5 × vip_handling_score
    Task 3 (Hard):   0.3 × time_efficiency     + 0.4 × relationship_health + 0.3 × priority_accuracy

CONNECTS TO:
    ← inference.py   (passes episode_history and final_state here after episode ends)
    ← client.py      (get_state() provides final State for relationship scoring)
    ← models.py      (State, Email, Relationship types)

RUN:
    from grader import grade_episode
    score = grade_episode(episode_history, final_state, task_id=1)

OWNER: LLM Engineer
"""

from typing import List, Dict, Any
from models import State, Email, Relationship


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def grade_episode(episode_history: List[dict], final_state: State, task_id: int) -> float:
    """
    Grade a completed episode.

    Args:
        episode_history: List of step dicts from inference.py
                         Each dict: {step, observation, action, reward, info}
        final_state:     State object from client.get_state() at episode end
        task_id:         1 (easy), 2 (medium), or 3 (hard)

    Returns:
        float: Score between 0 and 100.
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

    Score = 0.4 × (total_value_gained / max_possible_value)
          + 0.6 × (avg_relationship_health / 100)

    Rewards: Getting through the inbox efficiently while keeping relationships healthy.
    """
    value_score = calculate_value_efficiency(episode_history, final_state)
    rel_score = calculate_avg_relationship_health(final_state)

    score = 0.4 * value_score + 0.6 * rel_score
    return round(score * 100, 2)


def grade_task_2(episode_history: List[dict], final_state: State) -> float:
    """
    Task 2: VIP Tracking

    Score = 0.5 × correct_priority_rate
          + 0.5 × vip_handling_score

    vip_handling_score measures:
        - Did the agent respond to VIP emails?
        - Did it avoid ignoring VIPs repeatedly?
        - Did follow-up VIP emails (is_followup=True) get handled?
    """
    priority_score = calculate_priority_accuracy(episode_history)
    vip_score = calculate_vip_handling_score(episode_history, final_state)

    score = 0.5 * priority_score + 0.5 * vip_score
    return round(score * 100, 2)


def grade_task_3(episode_history: List[dict], final_state: State) -> float:
    """
    Task 3: Full Relationship Management

    Score = 0.3 × time_efficiency
          + 0.4 × relationship_health
          + 0.3 × priority_accuracy

    time_efficiency: How well did agent use its time budget?
                     (responded to high-value emails, ignored low-value ones)
    relationship_health: Average final health across all senders.
    priority_accuracy: Did agent respond to high-priority and ignore low-priority?
    """
    time_score = calculate_time_efficiency(episode_history, final_state)
    rel_score = calculate_avg_relationship_health(final_state)
    priority_score = calculate_priority_accuracy(episode_history)

    score = 0.3 * time_score + 0.4 * rel_score + 0.3 * priority_score
    return round(score * 100, 2)


# ---------------------------------------------------------------------------
# METRIC CALCULATORS
# ---------------------------------------------------------------------------

def calculate_value_efficiency(episode_history: List[dict], final_state: State) -> float:
    """
    Measures: Did the agent respond to high-value emails and ignore low-value ones?

    Formula: total_reward_earned / max_possible_reward
    max_possible_reward = sum of rewards if agent responded to all priority >= 7 emails
                          and ignored all priority <= 3 emails

    Returns float in [0, 1].
    """
    total_reward = sum(step["reward"] for step in episode_history)

    # Estimate max possible reward (respond to everything priority >= 5)
    max_possible = 0.0
    for email in final_state.inbox:
        if email.base_priority >= 5:
            max_possible += email.base_priority * 0.5  # Simplified estimate

    if max_possible <= 0:
        return 0.5  # Can't calculate, neutral score

    ratio = total_reward / max_possible
    return min(1.0, max(0.0, ratio))


def calculate_avg_relationship_health(final_state: State) -> float:
    """
    Average relationship health across all senders at episode end.
    Weights VIP relationships 3× more than Normal, ignores Spam.

    Returns float in [0, 1].
    """
    if not final_state.relationships:
        return 0.5

    weighted_sum = 0.0
    weight_total = 0.0

    for sender, rel in final_state.relationships.items():
        if rel.importance == "Spam":
            continue  # Spam relationships don't count
        weight = rel.importance_weight  # VIP=3, Normal=2
        weighted_sum += rel.health * weight
        weight_total += weight * 100  # Max possible contribution

    if weight_total <= 0:
        return 0.5

    return weighted_sum / weight_total


def calculate_priority_accuracy(episode_history: List[dict]) -> float:
    """
    Measures: Did the agent make correct RESPOND/IGNORE decisions based on priority?

    Correct decision:
        - High priority (>= 7) + VIP → should RESPOND (action=1)
        - Low priority (<= 3) + Spam → should IGNORE (action=0)
        - Everything else → partial credit

    Returns float in [0, 1].
    """
    if not episode_history:
        return 0.0

    correct = 0
    total = len(episode_history)

    for step in episode_history:
        obs = step["observation"]
        action = step["action"]
        importance = obs.get("sender_importance", "Normal")

        # Clear correct: respond to VIP high-priority
        if importance == "VIP" and action == 1:
            correct += 1
        # Clear correct: ignore spam
        elif importance == "Spam" and action == 0:
            correct += 1
        # Partial: anything else we give half credit
        else:
            correct += 0.5

    return correct / total


def calculate_vip_handling_score(episode_history: List[dict], final_state: State) -> float:
    """
    Measures specifically how well the agent handled VIP senders.

    Components:
        - VIP response rate: % of VIP emails that were responded to
        - VIP health: average health of VIP relationships at end
        - Follow-up penalty: points deducted for each VIP follow-up that was generated
                             (follow-ups mean we failed a previous VIP interaction)

    Returns float in [0, 1].
    """
    vip_steps = [s for s in episode_history if s["observation"].get("sender_importance") == "VIP"]

    if not vip_steps:
        return 1.0  # No VIPs → not applicable, give full score

    # VIP response rate
    vip_responds = sum(1 for s in vip_steps if s["action"] == 1)
    vip_response_rate = vip_responds / len(vip_steps)

    # VIP relationship health at end
    vip_health_scores = [
        rel.health / 100
        for rel in final_state.relationships.values()
        if rel.importance == "VIP"
    ]
    avg_vip_health = sum(vip_health_scores) / len(vip_health_scores) if vip_health_scores else 0.5

    # Follow-up penalty (emails in inbox with is_followup=True)
    followup_count = sum(1 for e in final_state.inbox if e.is_followup)
    followup_penalty = min(0.3, followup_count * 0.05)  # Max -0.3 penalty

    score = 0.5 * vip_response_rate + 0.5 * avg_vip_health - followup_penalty
    return min(1.0, max(0.0, score))


def calculate_time_efficiency(episode_history: List[dict], final_state: State) -> float:
    """
    Measures: Did the agent use its time budget wisely?

    Good efficiency = high-value emails responded to, time not wasted on spam.
    Bad efficiency = spent 2 hours on spam, ran out of time for VIP emails.

    Formula: value_per_minute_spent vs theoretical_max_value_per_minute

    Returns float in [0, 1].
    """
    total_time_spent = final_state.total_time_spent
    if total_time_spent <= 0:
        return 0.5  # No time spent — neutral

    # Total reward earned
    total_reward = sum(step["reward"] for step in episode_history)

    # Normalize by time spent (reward per minute)
    reward_per_minute = total_reward / total_time_spent

    # Benchmark: a "perfect" agent would earn ~1 reward point per minute on average
    # (tune this based on your email_bank values)
    BENCHMARK_REWARD_PER_MINUTE = 1.0
    efficiency = reward_per_minute / BENCHMARK_REWARD_PER_MINUTE

    return min(1.0, max(0.0, efficiency))


# ---------------------------------------------------------------------------
# SUMMARY PRINTER
# ---------------------------------------------------------------------------

def print_score_report(episode_history: List[dict], final_state: State, task_id: int):
    """Print a human-readable score report after an episode."""
    score = grade_episode(episode_history, final_state, task_id)

    print(f"\n{'='*50}")
    print(f"  TASK {task_id} SCORE REPORT")
    print(f"{'='*50}")
    print(f"  Final Score:      {score:.1f} / 100")

    rel_score = calculate_avg_relationship_health(final_state)
    priority_score = calculate_priority_accuracy(episode_history)

    print(f"\n  Avg Relationship Health: {rel_score*100:.1f}%")
    print(f"  Priority Accuracy:       {priority_score*100:.1f}%")

    print(f"\n  VIP Relationships:")
    for sender, rel in final_state.relationships.items():
        if rel.importance == "VIP":
            status = "✓" if rel.health >= 60 else "✗"
            print(f"    {status} {sender}: {rel.health:.0f}/100")

    print(f"{'='*50}\n")
    return score
