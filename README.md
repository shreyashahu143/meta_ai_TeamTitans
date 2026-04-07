# 📧 Email Triage RL Environment

> **Meta AI Hackathon — Team Titans**  
> An OpenEnv-compatible reinforcement learning environment for email inbox management.

---

## 🧠 What Problem Are We Solving?

Most email management tools rely on static priority labels — simple scores that never change. But real-world email inboxes are **dynamic, sequential decision-making environments** where every action has consequences over time.

Current systems miss two critical blind spots:

### ⚡ Blind Spot 1 — Action Cost Asymmetry
Not every email costs the same time and effort. A quick "Got it" reply takes 5 minutes, while reviewing a 50-page audit report can take 3 hours. An intelligent agent must learn to ask:
> *"Is the expected reward of this email worth the actual time cost right now?"*

### 👥 Blind Spot 2 — Sender Relationship Decay
The cost of ignoring someone is **not fixed** — it compounds with every delay. Ignoring your boss or a key client once may be tolerable, but repeated inaction:
- Damages the relationship (health drops from 75 → 55 → 35 → 0)
- Triggers escalating follow-up emails injected directly into the live inbox
- Reduces the value of future interactions with that sender

---

## 🧪 Environment Design: The "Learnable Pattern" (Partial Observability)

To score highly on the OpenEnv rubric for **Real-World Utility** and **Environment Design**, we avoid hardcoded, unlearnable traps. Instead, we built a mathematically sound **partial observability** mechanic.

In a real inbox, you do not know exactly how many minutes an email will take to resolve until you actually do it. However, you can visually estimate the burden based on the length of the email. We model this explicitly to test an RL agent's ability to learn proxy signals.

### How it works under the hood

There are two distinct variables in our environment:

1. `email_length`: The character count of the email body (e.g., `280 chars`). **This is what the agent actually sees in its observation.**
2. `estimated_response_time`: The true cost in minutes to handle the email. **The agent NEVER sees this directly.**

If we hardcoded `estimated_response_time` independently in a JSON file, the environment would be random noise. An agent might see a 280‑character email cost 180 minutes one episode and 5 minutes the next. **An RL agent cannot learn from random noise.**

### The fix & the mechanic

Instead of hardcoding, our environment dynamically computes the hidden `estimated_response_time` strictly from the observable `email_length` using a bucketed mathematical function:

| Email length (chars) | Estimated response time (minutes) |
|----------------------|-----------------------------------|
| < 150                | 5–15 (quick reply)                |
| 150–300              | 15–35 (standard)                  |
| 300–500              | 35–70 (deep work)                 |
| > 500                | 70–120 (black hole)               |

### Why this genuinely challenges frontier models

This creates a **true learnable pattern**. Over multiple episodes, the agent will learn: *"When I see a high `email_length`, responding will drain my `time_budget` massively."*

In **Task 2 (Pure Time Optimization)**, there are no VIP labels to guide the agent, and the time budget is severely restricted. A naive "always respond" agent will hit a 500‑character email, burn 90 minutes, and instantly fail. To succeed, the frontier model (e.g., Qwen2.5-72B) must learn to look at the observable proxy (`email_length`) and strategically ignore long emails to survive the time pressure.

---

## 🏗️ Architecture (The Restaurant Analogy)
OpenEnv = A Restaurant

server/environment.py ← The Kitchen (all game logic)
server/app.py ← The Pass (HTTP wrapper)
client.py ← The Waiter (agent ↔ server bridge)
models.py ← The Menu (type-safe contracts)
inference.py ← The Customer (LLM decision maker)
grader.py ← The Food Critic (scores the episode)
train_agents.py ← The Trainer (3 agent baselines)



### Request Flow
inference.py → client.py → POST /step → server/app.py → environment.py
↑ ↓
└──────────────── StepResponse(obs, reward, done, info) ───────────┘



---

## 📁 Project Structure
meta_ai_TeamTitans/
│
├── inference.py ← LLM agent (OpenEnv compliant, OpenAI client)
├── grader.py ← Scoring logic for all 3 tasks
├── client.py ← HTTP client (reset_env, step_env, get_state)
├── models.py ← Pydantic data contracts
├── train_agents.py ← Trains SMART / MEDIUM / DUMB baselines
├── test_agent.py ← Quick rule-based smoke test
├── test_Server.py ← API endpoint tests
├── validate-submission.sh ← Pre-submission validator
├── requirements.txt
├── Dockerfile
├── openenv.yaml
├── README.md
│
├── server/
│ ├── environment.py ← Core RL logic (reset, step, state)
│ ├── app.py ← FastAPI server (port 7860)
│ └── init.py
│
├── data/
│ └── email_bank.json ← 60 email templates (18 VIP, 32 Normal, 10 Spam)
│
└── tasks/
├── task_1_easy.json
├── task_2_medium.json
└── task_3_hard.json



---

## 🎮 Environment Details

### Observation Space (what the agent sees per step)

| Field | Type | Description |
|-------|------|-------------|
| `email_id` | int | Unique email identifier |
| `sender` | str | Sender email address |
| `subject` | str | Email subject line |
| `body` | str | Full email body |
| `sender_importance` | str | `VIP` / `Normal` / `Spam` |
| `email_length` | int | Character count — **proxy for time cost** (agent does NOT see actual minutes) |
| `relationship_score` | float | Current health with this sender only (0–100) |
| `time_budget_remaining` | int | Minutes left in workday |
| `emails_remaining` | int | How many emails are left |

> **Partial Observability:** The agent sees ONE email at a time. It does NOT see actual time costs, other senders' relationship scores, future emails, or relationship degradation rates.

### Action Space

| Action | Value | Effect |
|--------|-------|--------|
| IGNORE | `0` | Saves time. Relationship health decreases. If VIP: triggers follow-up injection. |
| RESPOND | `1` | Costs time proportional to email length. Relationship health increases (capped at 100). |

### Reward Function

**RESPOND (action = 1):**
email_value = base_priority × urgency_multiplier
normalized_cost = estimated_response_time / 120.0
reward = (email_value - 5.0 × normalized_cost) × (relationship_health / 100)



**IGNORE (action = 0):**
if Spam: reward = 0 (correct decision, no penalty)
else: health_penalty = 15 × sender_importance_weight
reward = -1 × (email_value × health_penalty / 100)



**Episode End — Sunset Penalty** (time ran out, emails remain):
penalty = -Σ(base_priority × relationship_health / 100) for all remaining emails



**Episode End — Time Bonus** (cleared entire inbox before time ran out):
bonus = (time_remaining / original_time_budget) × 10



### Relationship Decay Table

| Sender Type | Ignore Penalty | Respond Boost | Importance Weight |
|-------------|----------------|---------------|-------------------|
| VIP | −20 health | +15 health | 3 |
| Normal | −10 health | +10 health | 2 |
| Spam | 0 | 0 | 0 |

> **Repeat ignore extra penalty:** Ignoring the same sender more than once deducts an additional −10 health.

### Dynamic Follow-up Injection

When a VIP is ignored, a follow-up email is **dynamically inserted** into the live inbox 1–2 positions ahead:

- **1st ignore:** `"FOLLOW UP: {original subject}"` — priority +2, response time 30 min
- **2nd ignore:** `"URGENT — SECOND FOLLOW UP: {original subject}"` — priority +4 (capped at 10), extra −10 health

This makes Task 2 genuinely hard: a cascade of VIP follow-ups can overwhelm the inbox.

---

## 📊 Tasks

### Task 1 — Easy: Basic Prioritization

| Config | Value |
|--------|-------|
| Emails | 20 (5 VIP, 10 Normal, 5 Spam) |
| Time Budget | 480 min (8 hours — generous) |
| Avg min/email | 24 min |
| Time Pressure | Low — agent can finish all emails |

**Scoring:** `0.4 × value_efficiency + 0.6 × relationship_health`

**What the agent must learn:**
- Respond to VIP emails (ignoring hurts relationship health)
- Ignore Spam (responding wastes time with zero reward)
- Short high-priority Normal emails are worth responding to
- Very long low-priority Normal emails may not be worth the time cost

---

### Task 2 — Medium: VIP Relationship Tracking

| Config | Value |
|--------|-------|
| Emails | 25 (8 VIP, 12 Normal, 5 Spam) |
| Time Budget | 420 min (tighter) |
| Avg min/email | 16.8 min |
| Time Pressure | Medium — must skip some Normal/Spam |

**Scoring:** `0.5 × priority_accuracy + 0.5 × vip_handling_score`

**VIP Handling Score components:**
- 50% VIP response rate
- 50% average VIP relationship health at end
- −0.05 per follow-up email in inbox (max −0.30)

**What the agent must learn:**
- Never ignore VIPs — each ignore injects a follow-up with higher priority
- Follow-ups arrive 1–2 steps later, compounding the problem
- Must budget time carefully: skip Normal/Spam to protect time for VIPs

---

### Task 3 — Hard: Full Relationship Management

| Config | Value |
|--------|-------|
| Emails | 30 (8 VIP, 14 Normal, 8 Spam) |
| Time Budget | 360 min (6 hours — severe) |
| Avg min/email | 12 min |
| Time Pressure | Severe — cannot finish all emails |

**Scoring:** `0.3 × time_efficiency + 0.4 × relationship_health + 0.3 × priority_accuracy`

**Time efficiency benchmark:** Compares actual reward to a perfect greedy agent (picks highest-priority emails first until time runs out).

**What the agent must learn:**
- **Cannot respond to everything** — must choose what to skip
- Time-trap emails (120–180 min) from VIPs may NOT be worth responding to if time is low
- Ignoring a VIP twice triggers a second, more urgent follow-up + extra −10 health
- Sunset penalty punishes leaving high-priority emails unfinished
- Optimal policy: respond to short high-priority emails; skip long low-priority ones regardless of sender tier

---

## 🤖 Agent Baselines

Run `python train_agents.py` to train and compare 3 agent types:

| Agent | Strategy | Expected Task 1 | Expected Task 2 | Expected Task 3 |
|-------|----------|-----------------|-----------------|-----------------|
| **SMART** | VIP always, Spam never, Normal by cost/time | ~0.78 | ~0.82 | ~0.71 |
| **MEDIUM** | VIP always, Spam never, Normal random 60/40 | ~0.68 | ~0.65 | ~0.58 |
| **DUMB** | Always ignore everything | ~0.42 | ~0.38 | ~0.35 |

The SMART > MEDIUM > DUMB gap proves the grader is sensitive to agent behavior (variance check).

---

## 🚀 Running the Project

### Prerequisites

- Python 3.11
- `pip install -r requirements.txt`
- A `.env` file in the project root (see below)

### Environment Variables (.env)

```env
HF_TOKEN=your_huggingface_or_openai_key
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
TASK_ID=1
ENV_SERVER_URL=http://localhost:7860
Step 1 — Start the server
bash
# From project root
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
Step 2 — Quick smoke test (no API key needed)
bash
python test_agent.py
Step 3 — Run the LLM agent
bash
# Task 1
python inference.py

# Task 2
set TASK_ID=2 && python inference.py   # Windows
TASK_ID=2 python inference.py          # Mac/Linux

# Task 3
set TASK_ID=3 && python inference.py
Step 4 — Run baseline training
bash
pip install matplotlib
python train_agents.py
# Produces: reward_convergence_task1.png, all_agents_all_tasks.png, training_results.json
Step 5 — API manual tests
bash
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"action": 1}'
curl http://localhost:7860/state
🐳 Docker
bash
# Build
docker build -t email-triage-env .

# Run
docker run -p 7860:7860 email-triage-env

# Test
curl http://localhost:7860/health
✅ Pre-Submission Validation
bash
chmod +x validate-submission.sh
./validate-submission.sh https://your-team.hf.space
Checks: HF Space live, mandatory files present, env vars in inference.py, log markers, Docker build, openenv validate.

📐 OpenEnv Spec Compliance
Requirement	Status
reset() / step() / state() endpoints	✅
openenv.yaml present and valid	✅
3+ tasks with distinct scoring formulas	✅
Grader returns score in [0.0, 1.0]	✅
inference.py uses OpenAI client	✅
inference.py reads API_BASE_URL, MODEL_NAME, HF_TOKEN	✅
[START] / [STEP] / [END] stdout logging	✅
Dockerfile builds and runs on port 7860	✅
Deep copy in state()	✅
🛡️ validate-submission.sh — What It Is and Why It Exists
The competition judges in Phase 1 run a fully automated robot that pings your Hugging Face Space and checks a list of requirements. If Phase 1 fails, you are disqualified immediately — no human ever looks at your project, no second chance.

validate-submission.sh is a practice run of that exact robot, so you catch disqualifying issues yourself before submitting.

What it checks and why each one matters
Check	What it does	Why it matters
1. HF Space ping	Sends POST /reset to your Space URL	If your Space is down = instant DQ
2. Mandatory files	Checks inference.py, openenv.yaml, requirements.txt, Dockerfile exist	Missing even one = DQ
3. Env vars in inference.py	Checks API_BASE_URL, MODEL_NAME, HF_TOKEN are referenced	Judges inject these vars when running your script — if your code ignores them, score = 0
4. Log markers	Checks [START], [STEP], [END] exist in inference.py	Judges parse stdout to extract your score — wrong format = score recorded as 0
5. Docker build	Runs docker build on your repo	If it doesn't build, it doesn't deploy = DQ
6. openenv validate	Runs the OpenEnv framework validator	Checks your openenv.yaml matches the spec
How to run it
bash
# macOS / Linux
chmod +x validate-submission.sh
./validate-submission.sh https://your-team.hf.space

# Windows (run inside Git Bash, not PowerShell)
bash validate-submission.sh https://your-team.hf.space
All checks must say PASSED before you submit. Run this every time before pushing a final version.

⚠️ Troubleshooting — .env Not Loading
If you see error=Error code: 401 and your_api*****here in the output, your .env file is not being read correctly. Here's how to diagnose:

Check what Python is actually loading:

powershell
# Windows PowerShell
Get-Content .env
bash
# macOS / Linux
cat .env
If it still shows your_api_key_here, your file edit didn't save. Fix and save again.

If the file looks correct but it's still not working, set variables directly in the terminal:

powershell
# Windows PowerShell — run these, then immediately run inference.py WITHOUT closing the terminal
$env:HF_TOKEN = "hf_yourrealtokenhere"
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
python inference.py
bash
# macOS / Linux
export HF_TOKEN="hf_yourrealtokenhere"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
python inference.py
To verify what token Python loaded, add this temporarily to inference.py after load_dotenv():

python
import os
print("TOKEN:", os.getenv("HF_TOKEN"))
print("URL:", os.getenv("API_BASE_URL"))
Run once, check the output, then remove those lines.

Getting your real HF token
Go to huggingface.co and log in

Click your profile picture → Settings → Access Tokens

Click New Token → name it anything → Role: Read → Create

Copy the token — it starts with hf_

Paste it into .env as HF_TOKEN=hf_yourtoken

⚠️ Never commit your .env file. Make sure .env is listed in .gitignore.

✅ Pre-Submission Checklist
Run the validator first:

bash
./validate-submission.sh https://your-team.hf.space
Then verify manually:

Real hf_ token in .env (not the placeholder)

API_BASE_URL=https://router.huggingface.co/v1 in .env

Server starts without errors: uvicorn app:app --port 7860

curl http://localhost:7860/health returns {"status": "ok"}

python inference.py completes and prints [END] line with score=

No 401 errors in the [STEP] lines

All three tasks run cleanly (TASK_ID=1, 2, 3)

All scores are in [0.0, 1.0] range

.env is in .gitignore and does NOT appear in git status

docker build . succeeds locally

🐙 Git Workflow
bash
# First time setup
git init
git add .
git commit -m "initial project structure"
git remote add origin https://github.com/YOUR_USERNAME/meta_ai_TeamTitans.git
git branch -M main
git push -u origin main

# Daily workflow
git pull                         # always pull before starting work
# ... make changes ...
git add .
git commit -m "describe what you changed"
git push
⚠️ Always check git status before committing. .env must NEVER appear in the list.

🧪 stdout Log Format (mandatory per spec)
text
[START] task=basic-prioritization env=email-triage-rl model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=RESPOND reward=3.75 done=false error=null
[STEP] step=2 action=IGNORE reward=-4.20 done=false error=null
...
[END] success=true steps=20 score=0.743 rewards=3.75,-4.20,...
Any deviation from this format will cause incorrect evaluator scoring.
