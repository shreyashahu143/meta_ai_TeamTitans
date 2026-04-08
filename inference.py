"""
inference.py — LLM Agent (OpenEnv Compliant)
=============================================

Meta OpenEnv Hackathon 2026 — FINAL COMPLIANT VERSION

MANDATORY per competition spec:
    - Uses OpenAI client only (no Anthropic, no LangChain, no direct HTTP)
    - Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment
    - Emits [START], [STEP], [END] with exact field order and formatting
    - [END] is always emitted even on exception (try-finally)
    - Score formatted to 3 decimal places (matches official sample script log_end)

HOW TO RUN:
    export HF_TOKEN=your_hf_token_here
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
# MANDATORY ENV VARS (per OpenEnv spec — DO NOT rename these variables)
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    # Must emit valid [END] before exiting so evaluator logs are clean
    print("[END] success=false steps=0 score=0.000 rewards=", flush=True)
    sys.exit(1)

TASK_ID       = int(os.getenv("TASK_ID", "1"))
ENV_BENCHMARK = "email-triage-rl"

# ---------------------------------------------------------------------------
# TASK REGISTRY
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

# ---------------------------------------------------------------------------
# OPENAI CLIENT (mandatory — points to HuggingFace router, free tier)
# ---------------------------------------------------------------------------

openai_client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def load_task_config(task_id: int) -> dict:
    """Load JSON task config if file exists, else return empty dict."""
    path = TASK_CONFIGS.get(task_id, "")
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def build_prompt(obs) -> str:
    """
    Convert EmailObservation into a natural language prompt.

    Intentional partial observability:
      - Agent sees email_length hint (NOT raw time cost)
      - Agent sees relationship_score for THIS sender only
      - Agent does NOT see future emails in the queue
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
    """
    Extract 0 or 1 from LLM response.
    Defaults to RESPOND (1) on parse failure — conservative safe default.
    """
    text = llm_response.strip()
    if text in ("0", "1"):
        return int(text)
    match = re.search(r"\b([01])\b", text)
    if match:
        return int(match.group(1))
    return 1  # conservative default: respond rather than silently ignore


# ---------------------------------------------------------------------------
# STDOUT LOGGING — exact format required by spec
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    """[START] emitted once at episode begin."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error) -> None:
    """
    [STEP] emitted immediately after every env.step() call.
    - reward: 2 decimal places
    - done:   lowercase boolean (true/false)
    - error:  raw string or 'null'
    """
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    """
    [END] emitted after env.close(), always, even on exception.

    SPEC-CORRECT field order (official sample script, PDF p.13-15):
        [END] success=<bool> steps=<n> score=<score> rewards=<r1,r2,...,rn>

    score uses .3f (3 decimal places).
    """
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# MAIN EPISODE RUNNER
# ---------------------------------------------------------------------------

def run_episode(task_id: int = 1) -> dict:
    """
    Run one complete episode against the email-triage environment.
    Emits [START], one [STEP] per email, then [END] per OpenEnv spec.

    Uses try-finally to guarantee [END] is always emitted, even on crash.
    """
    import client

    # Grader is optional during development — fallback prevents import crash.
    # IMPORTANT: A real grader is required to pass Phase 2 variance checks.
    try:
        from grader import grade_episode
        has_grader = True
    except ImportError:
        has_grader = False

    task_name   = TASK_NAMES.get(task_id, f"task-{task_id}")
    task_config = load_task_config(task_id)

    log_start(task=task_name, env=ENV_BENCHMARK, model=MODEL_NAME)

    # Episode state — initialised before try so finally can always read them
    total_reward    = 0.0
    step_count      = 0
    rewards_log     = []
    episode_history = []
    success         = True
    final_state     = None
    score           = 0.0

    try:
        # --- Verify server is reachable before starting (safe check) ---
        # The spec does not require health_check, but it's helpful for debugging.
        # We use hasattr to avoid AttributeError if client lacks this method.
        if hasattr(client, "health_check") and not client.health_check():
            raise ConnectionError(
                "Server not running. Start with:\n"
                "  uvicorn server.app:app --host 0.0.0.0 --port 7860"
            )

        # --- Start episode ---
        obs = client.reset_env(task_config=task_config if task_config else None)

        # --- Episode loop: one email at a time ---
        while True:
            prompt    = build_prompt(obs)
            error_msg = None

            # LLM call — default to RESPOND (1) on any API failure
            try:
                response = openai_client.chat.completions.create(
                    model=MODEL_NAME,
                    max_tokens=10,
                    messages=[{"role": "user", "content": prompt}],
                )
                llm_response = response.choices[0].message.content or "1"
            except Exception as e:
                error_msg    = str(e)[:120]
                llm_response = "1"

            action       = parse_action(llm_response)
            action_label = "RESPOND" if action == 1 else "IGNORE"

            # Environment step
            try:
                result = client.step_env(action)
            except Exception as e:
                error_msg = str(e)[:120]
                success   = False
                # Log the failed step before re-raising so [STEP] count stays accurate
                log_step(
                    step   = step_count + 1,
                    action = action_label,
                    reward = 0.0,
                    done   = True,
                    error  = error_msg,
                )
                raise  # caught by outer except → finally emits [END]

            total_reward += result.reward
            step_count   += 1
            rewards_log.append(result.reward)

            # Mandatory [STEP] — emitted immediately after env.step()
            log_step(
                step   = step_count,
                action = action_label,
                reward = result.reward,
                done   = result.done,
                error  = error_msg,
            )

            # Use model_dump() for Pydantic v2 compatibility
            obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs.__dict__
            episode_history.append({
                "step":   step_count,
                "observation": obs_dict,
                "action":      action,
                "reward":      result.reward,
                "info":        result.info,
            })

            if result.done:
                break

            obs = result.observation

        # --- Grade episode ---
        final_state = client.get_state()

        if has_grader:
            score = grade_episode(episode_history, final_state, task_id)
        else:
            # Fallback: clip total_reward to [0,1] (will fail Phase 2 variance check)
            score = max(0.0, min(1.0, total_reward))

        score = float(max(0.0, min(1.0, score)))  # clamp to [0,1] per spec

    except Exception as e:
        success = False
        sys.stderr.write(f"[ERROR] Episode crashed: {e}\n")

    finally:
        # MANDATORY: [END] is ALWAYS emitted, even on exception
        log_end(
            success = success,
            steps   = step_count,
            score   = score,
            rewards = rewards_log,
        )
        # Clean up environment connection if client supports it
        if hasattr(client, "close_env"):
            try:
                client.close_env()
            except Exception:
                pass

    return {
        "task_id":         task_id,
        "total_reward":    total_reward,
        "steps":           step_count,
        "score":           score,
        "episode_history": episode_history,
        "final_state":     final_state,
    }


# ---------------------------------------------------------------------------
# ENTRY POINT — TASK_ID env var selects which task (default=1)
# Evaluators run all 3 tasks by changing TASK_ID externally.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_episode(task_id=TASK_ID)