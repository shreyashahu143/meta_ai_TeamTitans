"""
inference.py — LLM Agent (OpenEnv Compliant)
=============================================

WHAT THIS FILE DOES:
    Runs one complete episode of the Email Triage environment using an LLM agent.
    The LLM reads each email observation and decides: RESPOND (1) or IGNORE (0).

WHAT CHANGED FROM OLD VERSION:
    1. Switched from Anthropic SDK → OpenAI client (mandatory per spec)
    2. Reads API_BASE_URL, MODEL_NAME, HF_TOKEN env vars (mandatory per spec)
    3. Emits exact [START], [STEP], [END] log lines (mandatory per spec)
    4. Added env.close() concept (tracked via done flag)
    5. Removed argparse task selection — runs task 1 by default, configurable via env var

WHAT DID NOT CHANGE:
    - The prompt logic (build_prompt) — same reasoning, same partial observability
    - The action parser (parse_action) — same regex fallback logic
    - The episode loop structure — same reset → step → done flow
    - client.py calls — same reset_env, step_env, get_state
    - The grader call at the end — same episode_history format

OWNER: LLM Engineer
"""

import os
import sys
import json
import re

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# MANDATORY ENVIRONMENT VARIABLES (per OpenEnv spec)
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")     # Mandatory — no default

if HF_TOKEN is None:
    print("[END] success=false steps=0 rewards=", flush=True)
    raise ValueError("HF_TOKEN environment variable is required. Set it before running.")

# Task to run (override via TASK_ID env var, default=1)
TASK_ID = int(os.getenv("TASK_ID", "1"))

# ---------------------------------------------------------------------------
# TASK CONFIG LOADER
# ---------------------------------------------------------------------------

TASK_CONFIGS = {
    1: "tasks/task_1_easy.json",
    2: "tasks/task_2_medium.json",
    3: "tasks/task_3_hard.json",
}

TASK_NAMES = {
    1: "basic-prioritization",
    2: "vip-tracking",
    3: "full-relationship-management",
}

ENV_BENCHMARK = "email-triage-rl"


def load_task_config(task_id: int) -> dict:
    path = TASK_CONFIGS.get(task_id)
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# OPENAI CLIENT SETUP
# ---------------------------------------------------------------------------

openai_client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,       # HF_TOKEN is used as the API key per spec
)


# ---------------------------------------------------------------------------
# PROMPT BUILDER (unchanged from original — same logic, same partial observability)
# ---------------------------------------------------------------------------

def build_prompt(obs) -> str:
    """
    Convert EmailObservation into a natural language prompt.
    Agent sees: sender tier, subject, body, relationship score, time budget.
    Agent does NOT see: actual time cost, other senders' health, future emails.
    This partial observability is intentional — it's the core RL challenge.
    """
    if obs.email_length < 200:
        length_hint = "short (quick to handle)"
    elif obs.email_length < 800:
        length_hint = "medium length"
    else:
        length_hint = "long and detailed (may take significant time)"

    if obs.relationship_score >= 80:
        rel_hint = "excellent"
    elif obs.relationship_score >= 60:
        rel_hint = "good"
    elif obs.relationship_score >= 40:
        rel_hint = "strained"
    else:
        rel_hint = "critically damaged — this person is frustrated with you"

    return f"""You are an AI email triage assistant managing a professional inbox.

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

IMPORTANT TRADEOFFS:
- Ignoring VIPs damages your relationship and they WILL follow up more urgently
- Responding to spam wastes valuable time with no benefit
- Long emails cost more time — is this worth the investment given your remaining budget?
- If you run out of time, you will be penalized for all unfinished high-value work

Reply with ONLY a single digit: 0 or 1
Your decision:"""


# ---------------------------------------------------------------------------
# ACTION PARSER (unchanged — same regex fallback logic)
# ---------------------------------------------------------------------------

def parse_action(llm_response: str) -> int:
    """
    Extract 0 or 1 from LLM response.
    Handles edge cases: whitespace, 'Action: 1', 'I choose 0', etc.
    Defaults to RESPOND (1) if parsing fails — conservative default.
    """
    text = llm_response.strip()
    if text in ("0", "1"):
        return int(text)
    match = re.search(r"\b([01])\b", text)
    if match:
        return int(match.group(1))
    return 1  # Default: respond (better to over-respond than over-ignore)


# ---------------------------------------------------------------------------
# MAIN EPISODE RUNNER
# ---------------------------------------------------------------------------

def run_episode(task_id: int = 1) -> dict:
    """
    Run one complete episode using the LLM agent.
    Emits [START], [STEP], [END] log lines to stdout per OpenEnv spec.
    """
    import client

    task_name  = TASK_NAMES.get(task_id, f"task-{task_id}")
    task_config = load_task_config(task_id)

    # --- [START] line — emitted once at episode begin ---
    print(
        f"[START] task={task_name} env={ENV_BENCHMARK} model={MODEL_NAME}",
        flush=True
    )

    # --- Verify server ---
    if not client.health_check():
        print("[END] success=false steps=0 rewards=", flush=True)
        raise ConnectionError(
            "Server not running. Start with: uvicorn server.app:app --port 7860"
        )

    # --- Start episode ---
    obs = client.reset_env(task_config=task_config if task_config else None)

    total_reward    = 0.0
    step_count      = 0
    rewards_log     = []   # for [END] line
    episode_history = []   # for grader
    success         = True
    last_error      = None

    # --- Episode loop ---
    while True:
        # Build prompt and call LLM via OpenAI client
        prompt = build_prompt(obs)
        last_error = None

        try:
            response = openai_client.chat.completions.create(
                model=MODEL_NAME,
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}],
            )
            llm_response = response.choices[0].message.content
        except Exception as e:
            last_error = str(e)
            llm_response = "1"  # Default to RESPOND on error

        action       = parse_action(llm_response)
        action_label = "RESPOND" if action == 1 else "IGNORE"

        # Step the environment
        try:
            result = client.step_env(action)
        except Exception as e:
            last_error  = str(e)
            success     = False
            # Emit [END] immediately on env error
            rewards_str = ",".join(f"{r:.2f}" for r in rewards_log)
            print(
                f"[END] success=false steps={step_count} rewards={rewards_str}",
                flush=True
            )
            raise

        total_reward += result.reward
        step_count   += 1
        rewards_log.append(result.reward)

        # --- [STEP] line — emitted after every env.step() ---
        # Format: done=true/false (lowercase), reward=X.XX (2 decimal places)
        print(
            f"[STEP] step={step_count} "
            f"action={action_label} "
            f"reward={result.reward:.2f} "
            f"done={'true' if result.done else 'false'} "
            f"error={'null' if last_error is None else last_error}",
            flush=True
        )

        # Record for grader
        episode_history.append({
            "step":        step_count,
            "observation": obs.model_dump(),
            "action":      action,
            "reward":      result.reward,
            "info":        result.info,
        })

        if result.done:
            break

        obs = result.observation

    # --- env.close() equivalent — get final state ---
    final_state = client.get_state()

    # --- [END] line — always emitted, even on exception ---
    rewards_str = ",".join(f"{r:.2f}" for r in rewards_log)
    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={step_count} "
        f"rewards={rewards_str}",
        flush=True
    )

    return {
        "task_id":         task_id,
        "total_reward":    total_reward,
        "steps":           step_count,
        "episode_history": episode_history,
        "final_state":     final_state,
    }


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run_episode(task_id=TASK_ID)

    # Optional: print grader score after episode
    try:
        from grader import grade_episode
        score = grade_episode(result["episode_history"], result["final_state"], TASK_ID)
        print(f"\n[GRADE] task={TASK_ID} score={score:.4f}", flush=True)
    except Exception as e:
        print(f"\n[GRADE] error={e}", flush=True)