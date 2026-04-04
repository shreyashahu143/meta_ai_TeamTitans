"""
inference.py — LLM Agent (The Customer / Decision Maker)
=========================================================

PURPOSE:
    This is where the AI agent lives. It plays the email triage "game" by:
        1. Calling reset_env() to start a new episode
        2. Reading the EmailObservation (what email am I looking at?)
        3. Building a prompt and calling Claude API
        4. Parsing Claude's response → action (0 or 1)
        5. Calling step_env(action) to advance
        6. Repeating until done=True

    The LLM agent uses PARTIAL OBSERVABILITY — it only sees one email at a time,
    and does NOT see actual time costs or other senders' relationship scores.

CONNECTS TO:
    ← client.py          (imports reset_env, step_env, get_state)
    ← models.py          (EmailObservation type hints)
    → Anthropic API      (calls Claude to decide action)

SETUP:
    Set ANTHROPIC_API_KEY in your .env file or environment.
    Set ENV_SERVER_URL if server is not on localhost:8000.

RUN:
    python inference.py --task 1          # Run Task 1 (easy)
    python inference.py --task 2          # Run Task 2 (medium)
    python inference.py --task 3 --debug  # Run Task 3 with debug output

OWNER: LLM Engineer
"""

import argparse
import json
import os
import re
import sys

import anthropic
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()

import client
from models import EmailObservation

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = "claude-3-5-haiku-20241022"  # Fast and cheap for RL loops

# Task config files
TASK_CONFIGS = {
    1: "tasks/task_1_easy.json",
    2: "tasks/task_2_medium.json",
    3: "tasks/task_3_hard.json",
}


# ---------------------------------------------------------------------------
# PROMPT BUILDER
# ---------------------------------------------------------------------------

def build_prompt(obs: EmailObservation) -> str:
    """
    Convert EmailObservation into a natural language prompt for Claude.

    KEY DESIGN: We describe the email in plain English, NOT as JSON.
    The agent must reason about it like a human would.

    Hidden from agent (intentional partial observability):
        - Actual time cost (agent sees email_length as a proxy)
        - Other senders' relationship scores
        - Future emails
        - Degradation rates
    """
    # Translate length to rough difficulty hint
    if obs.email_length < 200:
        length_hint = "short (quick to handle)"
    elif obs.email_length < 800:
        length_hint = "medium length"
    else:
        length_hint = "long and detailed (may take significant time)"

    # Relationship health description
    if obs.relationship_score >= 80:
        rel_hint = "excellent"
    elif obs.relationship_score >= 60:
        rel_hint = "good"
    elif obs.relationship_score >= 40:
        rel_hint = "strained"
    else:
        rel_hint = "critically damaged — this person is frustrated with you"

    prompt = f"""You are an AI email triage assistant managing a professional inbox.

CURRENT SITUATION:
- Time remaining today: {obs.time_budget_remaining} minutes
- Emails left to process: {obs.emails_remaining}

CURRENT EMAIL:
- From: {obs.sender}
- Sender tier: {obs.sender_importance} (VIP = boss/key client, Normal = teammates, Spam = newsletters)
- Subject: {obs.subject}
- Email length: {length_hint}
- Your relationship with this sender: {rel_hint} ({obs.relationship_score:.0f}/100)

EMAIL BODY:
{obs.body}

DECISION:
You must choose one action:
  0 = IGNORE this email (saves time, but may damage the relationship — especially for VIPs)
  1 = RESPOND to this email (costs time proportional to email complexity, improves relationship)

IMPORTANT TRADEOFFS TO CONSIDER:
- Ignoring VIPs damages your relationship and they WILL follow up more urgently
- Responding to spam wastes valuable time with no benefit
- Long emails cost more time — is this worth the investment given your remaining budget?
- If you run out of time with emails left, you'll be penalized for all unfinished high-value work

Reply with ONLY a single digit: 0 or 1
Your decision:"""

    return prompt


# ---------------------------------------------------------------------------
# ACTION PARSER
# ---------------------------------------------------------------------------

def parse_action(llm_response: str) -> int:
    """
    Extract action (0 or 1) from LLM response.
    Handles edge cases: whitespace, "Action: 1", "I choose 0", etc.

    Returns:
        0 or 1. Defaults to 1 (respond) if parsing fails — conservative default.
    """
    text = llm_response.strip()

    # Direct digit
    if text in ("0", "1"):
        return int(text)

    # Find first digit in response
    match = re.search(r"\b([01])\b", text)
    if match:
        return int(match.group(1))

    # Fallback: respond (conservative — better to over-respond than over-ignore)
    print(f"[WARN] Could not parse action from: '{text[:100]}'. Defaulting to RESPOND (1).")
    return 1


# ---------------------------------------------------------------------------
# MAIN AGENT LOOP
# ---------------------------------------------------------------------------

def run_episode(task_id: int = 1, debug: bool = False) -> dict:
    """
    Run one complete episode using the LLM agent.

    Args:
        task_id: 1, 2, or 3. Loads the corresponding task config.
        debug: Print detailed step-by-step output.

    Returns:
        dict with episode summary: total_reward, steps, actions_taken, final_state
    """
    # --- Verify server is running ---
    if not client.health_check():
        raise ConnectionError(
            "Server is not running. Start it first:\n"
            "  cd server/ && uvicorn app:app --port 8000"
        )

    # --- Load task config ---
    task_config = {}
    task_path = TASK_CONFIGS.get(task_id)
    if task_path and os.path.exists(task_path):
        with open(task_path) as f:
            task_config = json.load(f)
        print(f"[INFO] Loaded task config: {task_path}")

    # --- Start episode ---
    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    print(f"\n{'='*50}")
    print(f"  EMAIL TRIAGE — TASK {task_id}")
    print(f"{'='*50}\n")

    obs = client.reset_env(task_config=task_config if task_config else None)

    total_reward = 0.0
    step_count = 0
    actions_taken = []
    episode_history = []

    # --- Episode loop ---
    while True:
        if debug:
            print(f"\n--- Step {step_count + 1} ---")
            print(f"Email: [{obs.sender_importance}] {obs.subject[:60]}")
            print(f"Relationship: {obs.relationship_score:.0f}/100 | Time left: {obs.time_budget_remaining} min | Emails left: {obs.emails_remaining}")

        # Build prompt and call LLM
        prompt = build_prompt(obs)

        try:
            message = anthropic_client.messages.create(
                model=MODEL,
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}],
            )
            llm_response = message.content[0].text
        except Exception as e:
            print(f"[ERROR] LLM call failed: {e}. Defaulting to RESPOND.")
            llm_response = "1"

        action = parse_action(llm_response)
        action_label = "RESPOND" if action == 1 else "IGNORE"

        if debug:
            print(f"Agent decision: {action_label} ({action})")

        # Step the environment
        result = client.step_env(action)
        total_reward += result.reward
        step_count += 1
        actions_taken.append(action)

        # Record history (for grader)
        episode_history.append({
            "step": step_count,
            "observation": obs.model_dump(),
            "action": action,
            "reward": result.reward,
            "info": result.info,
        })

        if debug:
            print(f"Reward: {result.reward:+.2f} | Total: {total_reward:+.2f}")

        if result.done:
            print(f"\n{'='*50}")
            print(f"  EPISODE COMPLETE")
            print(f"  Steps:        {step_count}")
            print(f"  Total Reward: {total_reward:+.2f}")
            print(f"  Responds:     {actions_taken.count(1)}")
            print(f"  Ignores:      {actions_taken.count(0)}")
            print(f"{'='*50}\n")
            break

        obs = result.observation

    # Get final state for grader
    final_state = client.get_state()

    return {
        "task_id": task_id,
        "total_reward": total_reward,
        "steps": step_count,
        "actions": actions_taken,
        "episode_history": episode_history,
        "final_state": final_state,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Email Triage LLM Agent")
    parser.add_argument("--task", type=int, default=1, choices=[1, 2, 3],
                        help="Task difficulty: 1=easy, 2=medium, 3=hard")
    parser.add_argument("--debug", action="store_true",
                        help="Print detailed step-by-step output")
    args = parser.parse_args()

    if not ANTHROPIC_API_KEY:
        print("[ERROR] ANTHROPIC_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    result = run_episode(task_id=args.task, debug=args.debug)
    print(f"Final result saved. Total reward: {result['total_reward']:+.2f}")
