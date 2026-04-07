# test_agent.py
import random
from client import reset_env, step_env, health_check, get_state
import sys
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# -----------------------------------------------------------------------
# SERVER CHECK
# -----------------------------------------------------------------------
if not health_check():
    print("ERROR: Server not running!")
    print("Start it with: uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload")
    exit(1)

print("Server is up. Starting test...\n")
print(f"{'Step':>4} | {'Action':>7} | {'Tier':>6} | {'Reward':>8} | {'Total':>9} | {'Time':>5} | {'Rel':>5} | Subject")
print("-" * 95)

# -----------------------------------------------------------------------
# EPISODE LOOP
# -----------------------------------------------------------------------
obs = reset_env()
total_reward = 0
step_count = 0
episode_log = []  # for summary at end

while step_count < 100:

    # --- Decision logic (rule-based placeholder for LLM agent) ---
    if obs.sender_importance == "VIP":
        action = 1                      # Always respond to VIP
    elif obs.sender_importance == "Spam":
        action = 0                      # Always ignore Spam
    else:
        action = random.choice([0, 1])  # Random for Normal

    result = step_env(action)
    total_reward += result.reward
    step_count += 1

    action_str = "RESPOND" if action == 1 else "IGNORE"
    subject_preview = obs.subject[:35].ljust(35) if hasattr(obs, "subject") else "—"

    print(
        f"{step_count:>4} | {action_str:>7} | {obs.sender_importance:>6} | "
        f"{result.reward:>8.2f} | {total_reward:>9.2f} | "
        f"{result.observation.time_budget_remaining:>5} | "
        f"{obs.relationship_score:>5.0f} | "
        f"{subject_preview}"
    )

    # Log for summary
    episode_log.append({
        "step": step_count,
        "action": action_str,
        "tier": obs.sender_importance,
        "reward": result.reward,
        "rel": obs.relationship_score,
    })

    if result.done:
        print(f"\n{'='*95}")
        print(f"  EPISODE ENDED at step {step_count}")
        print(f"  Final total reward : {total_reward:.2f}")

        # Check if time bonus was given
        if "time_bonus" in result.info:
            print(f"  Time bonus applied : +{result.info['time_bonus']:.2f}")
        if "sunset_penalty" in result.info:
            print(f"  Sunset penalty     : {result.info['sunset_penalty']:.2f}")
        print(f"{'='*95}")
        break

    obs = result.observation

if step_count >= 100:
    print(f"\nReached 100 steps without finishing! Total reward: {total_reward:.2f}")

# -----------------------------------------------------------------------
# RELATIONSHIP HEALTH SUMMARY  (calls /state after episode)
# -----------------------------------------------------------------------
try:
    final = get_state().model_dump()
    relationships = final.get("relationships", {})

    print("\n--- RELATIONSHIP HEALTH (end of episode) ---")
    print(f"  {'Tier':>6} | {'Sender':^40} | {'Health':>6} | Bar")
    print("  " + "-" * 70)

    for sender, rel in sorted(relationships.items(), key=lambda x: -x[1]["health"]):
        tier = rel["importance"]
        health = rel["health"]
        angry = " [ANGRY]" if rel.get("is_angry") else ""
        interactions = rel.get("interaction_count", 0)
        bar = "█" * int(health // 10) + "░" * (10 - int(health // 10))
        print(f"  {tier:>6} | {sender:<40} | {health:>5.1f}% | {bar}{angry}  (responded {interactions}x)")

    # Time budget summary
    time_remaining = final.get("time_budget_remaining", 0)
    time_spent = final.get("total_time_spent", 0)
    original_budget = time_remaining + time_spent
    pct_used = (time_spent / original_budget * 100) if original_budget > 0 else 0
    time_bonus_val = (time_remaining / original_budget) * 10 if original_budget > 0 else 0

    print(f"\n--- TIME BUDGET ---")
    print(f"  Used    : {time_spent} min / {original_budget} min  ({pct_used:.1f}%)")
    print(f"  Left    : {time_remaining} min")
    print(f"  Time bonus (if inbox cleared): +{time_bonus_val:.2f}")

    # Per-tier summary
    print(f"\n--- DECISION SUMMARY ---")
    for tier in ["VIP", "Normal", "Spam"]:
        tier_steps = [s for s in episode_log if s["tier"] == tier]
        responds = sum(1 for s in tier_steps if s["action"] == "RESPOND")
        ignores = sum(1 for s in tier_steps if s["action"] == "IGNORE")
        tier_reward = sum(s["reward"] for s in tier_steps)
        if tier_steps:
            print(f"  {tier:>6}: {responds} responded, {ignores} ignored | tier reward: {tier_reward:.2f}")

except Exception as e:
    print(f"\n[WARNING] Could not fetch final state: {e}")
    print("  Make sure get_state() is implemented in client.py")