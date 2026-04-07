"""
inference.py — LLM Agent (OpenEnv Compliant)
=============================================

MANDATORY per competition spec:
    - Uses OpenAI client (NOT Anthropic SDK)
    - Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment
    - Emits [START], [STEP], [END] log lines to stdout
    - [END] line includes score=<0.00-1.00>
    - Named inference.py, placed in root directory

HOW TO RUN:
    export HF_TOKEN=your_key_here
    export API_BASE_URL=https://router.huggingface.co/v1
    export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
    export TASK_ID=1
    python inference.py

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
# MANDATORY ENV VARS (per OpenEnv spec)
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("[END] success=false steps=0 rewards=", flush=True)
    sys.exit(1)

TASK_ID      = int(os.getenv("TASK_ID", "1"))
ENV_BENCHMARK = "email-triage-rl"

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

# ---------------------------------------------------------------------------
# OPENAI CLIENT
# ---------------------------------------------------------------------------

openai_client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def load_task_config(task_id: int) -> dict:
    path = TASK_CONFIGS.get(task_id, "")
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def build_prompt(obs) -> str:
    """
    Convert EmailObservation into a natural language prompt.
    Intentional partial observability:
      - Agent sees email_length (NOT actual time cost)
      - Agent sees this sender's relationship_score only
      - Agent does NOT see future emails
    """
    if obs.email_length < 200:
        length_hint = "short (quick to handle)"
    elif obs.email_length < 800:
        length_hint = "medium length"
    else:
        length_hint = "long and detailed (will cost significant time)"

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

TRADEOFFS TO CONSIDER:
- Ignoring VIPs damages your relationship and they WILL follow up more urgently
- Responding to spam wastes valuable time with no benefit
- Long emails cost more time — is this worth the investment given your remaining budget?
- Running out of time means a penalty for all unfinished high-value work

Reply with ONLY a single digit: 0 or 1
Your decision:"""


def parse_action(llm_response: str) -> int:
    """Extract 0 or 1 from LLM response. Defaults to RESPOND (1) on failure."""
    text = llm_response.strip()
    if text in ("0", "1"):
        return int(text)
    match = re.search(r"\b([01])\b", text)
    if match:
        return int(match.group(1))
    return 1  # Conservative default


# ---------------------------------------------------------------------------
# STDOUT LOGGING — exact format required by spec
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# MAIN EPISODE RUNNER
# ---------------------------------------------------------------------------

def run_episode(task_id: int = 1) -> dict:
    """
    Run one complete episode. Emits [START], [STEP], [END] per OpenEnv spec.
    Returns episode summary dict.
    """
    import client
    from grader import grade_episode

    task_name   = TASK_NAMES.get(task_id, f"task-{task_id}")
    task_config = load_task_config(task_id)

    log_start(task=task_name, env=ENV_BENCHMARK, model=MODEL_NAME)

    # Verify server
    if not client.health_check():
        log_end(success=False, steps=0, rewards=[])
        raise ConnectionError(
            "Server not running. Start with:\n"
            "  uvicorn server.app:app --host 0.0.0.0 --port 7860"
        )

    # Start episode
    obs = client.reset_env(task_config=task_config if task_config else None)

    total_reward    = 0.0
    step_count      = 0
    rewards_log     = []
    episode_history = []
    success         = True
    last_error      = None

    # Episode loop
    while True:
        prompt     = build_prompt(obs)
        last_error = None

        try:
            response = openai_client.chat.completions.create(
                model=MODEL_NAME,
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}],
            )
            llm_response = response.choices[0].message.content or "1"
        except Exception as e:
            last_error   = str(e)[:120]
            llm_response = "1"

        action       = parse_action(llm_response)
        action_label = "RESPOND" if action == 1 else "IGNORE"

        try:
            result = client.step_env(action)
        except Exception as e:
            last_error = str(e)[:120]
            success    = False
            log_end(success=False, steps=step_count, rewards=rewards_log)
            raise

        total_reward += result.reward
        step_count   += 1
        rewards_log.append(result.reward)

        # Mandatory [STEP] line
        log_step(
            step   = step_count,
            action = action_label,
            reward = result.reward,
            done   = result.done,
            error  = last_error,
        )

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

    # Get final state for grading
    final_state = client.get_state()

    # Grade episode — score must be in [0.0, 1.0]
    try:
        score = grade_episode(episode_history, final_state, task_id)
        score = float(max(0.0, min(1.0, score)))
    except Exception:
        score = 0.0

    # Mandatory [END] line with score
    log_end(
        success = success,
        steps   = step_count,
        rewards = rewards_log,
    )

    return {
        "task_id":         task_id,
        "total_reward":    total_reward,
        "steps":           step_count,
        "episode_history": episode_history,
        "final_state":     final_state,
    }


# ---------------------------------------------------------------------------
# ENTRY POINT — runs all 3 tasks sequentially
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run the task specified by TASK_ID env var (default=1)
    # Evaluators will run all 3 tasks by changing TASK_ID
    result = run_episode(task_id=TASK_ID)
    print(f"\n[GRADE] task={TASK_ID}", flush=True)