# test_agent.py
import random
from client import reset_env, step_env, health_check

# Check server is running
if not health_check():
    print("ERROR: Server not running!")
    print("Start it with: uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload")
    exit(1)

print("Server is up. Starting test...\n")

obs = reset_env()
total_reward = 0
step_count = 0

while step_count < 100:
    # Simple rules:
    if obs.sender_importance == "VIP":
        action = 1  # Always respond to VIP
    elif obs.sender_importance == "Spam":
        action = 0  # Always ignore Spam
    else:
        action = random.choice([0, 1])  # Random for Normal
    
    result = step_env(action)
    total_reward += result.reward
    step_count += 1
    
    action_str = "RESPOND" if action == 1 else "IGNORE"
    print(f"Step {step_count:2d}: {action_str} | {obs.sender_importance:6s} | reward={result.reward:7.2f} | total={total_reward:8.2f} | time_left={result.observation.time_budget_remaining}")
    
    if result.done:
        print(f"\n{'='*80}")
        print(f"EPISODE ENDED at step {step_count}")
        print(f"Final total reward: {total_reward:.2f}")
        print(f"{'='*80}")
        break
    
    obs = result.observation

if step_count >= 100:
    print("\nReached 100 steps without finishing!")
    print(f"Final total reward: {total_reward:.2f}")