# 📧 Email Triage RL Environment
### Team Titans — Meta AI Hackathon Submission

---

## 🧠 What Problem Are We Solving?

Most email management tools rely on static priority labels — simple scores that never change. But real-world inboxes are **dynamic, sequential decision-making environments** where every action has downstream consequences.

Current systems miss two critical blind spots:

**⚡ Action Cost Asymmetry**
Not every email costs the same time. A quick "Got it" takes 2 minutes; reviewing an 8-vendor contract audit takes 3 hours. An intelligent agent must ask: *"Is the expected reward worth the actual time cost right now?"*

**👥 Sender Relationship Decay**
Ignoring someone is not a one-time cost — it compounds. Ignore your boss once: tolerable. Three times: they're angry and sending escalating follow-ups. The agent must learn that relationship damage is an investment, not a fixed fee.

---

## 🏗️ Architecture (The Restaurant Analogy)

```
OpenEnv = A Restaurant

  server/environment.py  ← The Kitchen     (All game rules live here)
  server/app.py          ← The Pass        (FastAPI wrapper, no logic)
  client.py              ← The Waiter      (Moves data between AI and Kitchen)
  models.py              ← The Menu        (Type-safe contracts for everything)
  inference.py           ← The Customer    (The AI agent making decisions)
  grader.py              ← The Food Critic (Scores how well the AI did)
  data/email_bank.json   ← The Ingredients (82 pre-written email templates)
```

### Request Flow

```
inference.py  →  client.py  →  POST /step  →  server/app.py  →  environment.py
     ↑                                                                  |
     └──────────────── StepResponse (obs, reward, done, info) ─────────┘
```

---

## 📁 File Structure

```
meta_ai_TeamTitans/
│
├── inference.py           ← LLM agent (OpenAI client, mandatory per spec)
├── client.py              ← HTTP client connecting agent ↔ server
├── models.py              ← All data types (Email, State, Observation, Action)
├── grader.py              ← Scoring for Tasks 1, 2, 3
├── requirements.txt       ← Python dependencies
├── openenv.yaml           ← OpenEnv framework manifest
├── Dockerfile             ← Container (Python 3.11, port 7860)
├── README.md              ← This file
├── validate-submission.sh ← Pre-submission validation script
│
├── server/
│   ├── environment.py     ← Core RL logic: reset(), step(), state()
│   └── app.py             ← FastAPI endpoints: /reset, /step, /state, /health
│
├── data/
│   └── email_bank.json    ← 82 email templates (20 VIP, 46 Normal, 16 Spam)
│
├── tasks/
│   ├── task_1_easy.json   ← 20 emails, 480 min budget
│   ├── task_2_medium.json ← 25 emails, 420 min budget, VIP focus
│   └── task_3_hard.json   ← 30 emails, 360 min budget, full pressure
│
└── tests/
    ├── test_environment.py  ← Unit tests for environment logic
    ├── test_client.py       ← Integration tests (requires running server)
    └── test_grader.py       ← Unit tests for scoring functions
```

---

## 🐍 Setup — Python 3.11 with venv

<<<<<<< HEAD
> **Standard for this hackathon:** Python **3.11** with `venv`. Do NOT use conda or poetry .
>
> To check which Python versions you have installed, run:
> ```powershell
> py -0
> ```
> You need (3.11 preferred). If you don't have either, download from python.org.

### Step 1: Create the virtual environment

```powershell
# Windows (PowerShell) — use py launcher to pick the right version
py -3.11 -m venv venv
```

```bash
# macOS/Linux
python3.11 -m venv venv
=======
> **Required Python version: 3.11**
> Check what you have: `python --version` or `py -0` on Windows.
> Download 3.11 from [python.org](https://python.org) if needed.

### Step 1 — Create the virtual environment

```bash
# macOS / Linux
python3.11 -m venv venv

# Windows (PowerShell — use py launcher)
py -3.11 -m venv venv
>>>>>>> kezia
```

### Step 2 — Activate it

```bash
# macOS / Linux
source venv/bin/activate

# Windows PowerShell
venv\Scripts\Activate.ps1

# Windows Command Prompt
venv\Scripts\activate.bat
```

> ⚠️ **Windows permissions error?** Run this once, then try again:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

You'll know it's active when you see `(venv)` at the start of your terminal prompt.

<<<<<<< HEAD
```bat
:: Windows Command Prompt
venv\Scripts\activate.bat
```

### Step 3: Copy and fill in your .env file

```powershell
copy .env
```

```bash
# macOS/Linux
cp .env
```

Then open `.env` and paste in your `ANTHROPIC_API_KEY`.

### Step 4: Install dependencies

```powershell
=======
### Step 3 — Install dependencies

```bash
>>>>>>> kezia
pip install -r requirements.txt
```

### Step 4 — Set environment variables

<<<<<<< HEAD
```powershell
python --version   # Should show Python 3.11.x
pip list           # Should show fastapi, pydantic, anthropic, etc.
=======
Create a `.env` file in the project root (never commit this file):

```env
HF_TOKEN=your_huggingface_token_here
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
TASK_ID=1
ENV_SERVER_URL=http://localhost:7860
>>>>>>> kezia
```

### Step 5 — Verify

```bash
python --version    # Should show Python 3.11.x
pip list            # Should show fastapi, openai, pydantic, uvicorn, etc.
```

> ⚠️ **Always activate the venv before running any file in this project.**

---

## 🚀 Running the Project

### Option A — Run locally (recommended for development)

```bash
<<<<<<< HEAD
# Terminal 1 — Start the server
source venv/bin/activate
cd server/
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
=======
# Terminal 1: Start the server
source venv/bin/activate          # macOS/Linux
# venv\Scripts\Activate.ps1      # Windows
>>>>>>> kezia

cd server/
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

```bash
# Terminal 2: Run the agent
source venv/bin/activate
python inference.py               # Runs task defined by TASK_ID env var (default: 1)
```

To run all three tasks:
```bash
TASK_ID=1 python inference.py
TASK_ID=2 python inference.py
TASK_ID=3 python inference.py
```

### Option B — Run with Docker

```bash
# Build
docker build -t email-triage-env .

<<<<<<< HEAD
# Run the container
=======
# Run server
>>>>>>> kezia
docker run -p 7860:7860 email-triage-env

# In another terminal, run the agent
source venv/bin/activate
python inference.py
```

### Test the API manually

```bash
# Check server is up
curl http://localhost:7860/health

# Start a new episode
curl -X POST http://localhost:7860/reset

# Take an action (0 = ignore, 1 = respond)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": 1}'

# Get full internal state (god-mode view)
curl http://localhost:7860/state
```

### Run the test agent (rule-based, no LLM)

```bash
# Server must be running first
python test_agent.py
```

### Run unit tests

```bash
# Environment logic (no server needed)
pytest tests/test_environment.py -v

# Grader logic (no server needed)
pytest tests/test_grader.py -v

# Client ↔ Server integration (server must be running)
pytest tests/test_client.py -v
```

---

## 🎮 Environment Details

### Observation (what the agent sees per step)

| Field | Type | Description |
|---|---|---|
| `email_id` | int | Unique identifier |
| `sender` | str | Email address |
| `subject` | str | Subject line |
| `body` | str | Full body text |
| `sender_importance` | str | VIP / Normal / Spam |
| `email_length` | int | Character count — **proxy for time cost only** |
| `relationship_score` | float | Health with this sender (0–100) |
| `time_budget_remaining` | int | Minutes remaining in workday |
| `emails_remaining` | int | How many emails left to process |

> **Partial Observability:** The agent sees one email at a time. It does NOT see actual response time, other senders' relationship scores, or future emails.

### Action Space

| Action | Value | Effect |
|---|---|---|
| IGNORE | `0` | Time budget unchanged. Relationship health decreases. VIPs become angry. |
| RESPOND | `1` | Time budget decreases. Relationship health increases. |

### Reward Function

**RESPOND:**
```
email_value     = base_priority × urgency_multiplier
normalized_cost = estimated_response_time / 120.0
reward = (email_value - 5.0 × normalized_cost) × (relationship_health / 100)
```

**IGNORE:**
```
# Spam: reward = 0 (correct decision, no cost)
# Others:
health_penalty = 15 × importance_weight
reward = -1 × (email_value × health_penalty / 100)
```

**Episode End — Sunset Penalty** (time runs out):
```
penalty = -Σ(base_priority × relationship_health / 100)  for all remaining emails
```

**Episode End — Time Bonus** (inbox cleared with time remaining):
```
bonus = (time_remaining / original_budget) × 10
```

### Relationship Decay Table

| Sender Tier | Ignore Penalty | Respond Boost | Importance Weight |
|---|---|---|---|
| VIP | −20 health | +15 health | 3 |
| Normal | −10 health | +10 health | 2 |
| Spam | 0 | 0 | 0 |

---

## 📊 Tasks

### Task 1 — Easy: Basic Prioritization
- **Emails:** 20 (5 VIP, 10 Normal, 5 Spam)
- **Time Budget:** 480 minutes
- **Goal:** Respond to high-value emails, ignore spam
- **Score:** `0.4 × value_efficiency + 0.6 × relationship_health`

### Task 2 — Medium: VIP Relationship Tracking
- **Emails:** 25 (8 VIP, 12 Normal, 5 Spam)
- **Time Budget:** 420 minutes
- **Goal:** Learn that ignoring VIPs triggers escalating follow-up chains
- **Score:** `0.5 × priority_accuracy + 0.5 × vip_handling_score`

### Task 3 — Hard: Full Multi-Objective Optimization
- **Emails:** 30 (8 VIP, 14 Normal, 8 Spam)
- **Time Budget:** 360 minutes
- **Goal:** Balance time efficiency, priority accuracy, AND relationship health under pressure
- **Score:** `0.3 × time_efficiency + 0.4 × relationship_health + 0.3 × priority_accuracy`

All grader scores are in **[0.0, 1.0]** as required by OpenEnv spec.

---

## 📧 Email Bank Structure

`data/email_bank.json` contains **60 pre-written email templates** across three categories:

| Category | Count | ID Range | Key Features |
|---|---|---|---|
| VIP | 18 | 101–120 | 4 follow-up chains, 2 low-priority traps, 3 time-trap emails (120–180 min) |
| Normal | 32 | 201–232 | 3 follow-ups, 2 ambiguous-priority emails, 2 high-cost traps (90 min) |
| Spam | 10 | 301–310 | Mix of newsletters, promos, recruiters, webinars |

**Key design features:**
- Low-priority VIP emails (coffee order, venue poll) test whether the agent blindly responds to all VIPs
- Ambiguous normal emails (intern finds critical security bug) test content-reading over sender tier
- Follow-up chains with `parent_email_id` test relationship decay awareness
- Time-trap emails (180-min audit) test cost-awareness — even from VIPs

---

## 🔑 Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | ✅ Yes | — | Hugging Face API token for LLM calls |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `TASK_ID` | No | `1` | Which task to run (1, 2, or 3) |
| `ENV_SERVER_URL` | No | `http://localhost:7860` | Where to find the environment server |

---

## 🛡️ validate-submission.sh — What It Is and Why It Exists

<<<<<<< HEAD
```env
ANTHROPIC_API_KEY=your_key_here
ENV_SERVER_URL=http://localhost:7860
TASK_ID=1
=======
The competition judges in **Phase 1 run a fully automated robot** that pings your Hugging Face Space and checks a list of requirements. If Phase 1 fails, you are **disqualified immediately** — no human ever looks at your project, no second chance.

`validate-submission.sh` is a **practice run of that exact robot**, so you catch disqualifying issues yourself before submitting.

### What it checks and why each one matters

| Check | What it does | Why it matters |
|---|---|---|
| 1. HF Space ping | Sends `POST /reset` to your Space URL | If your Space is down = instant DQ |
| 2. Mandatory files | Checks `inference.py`, `openenv.yaml`, `requirements.txt`, `Dockerfile` exist | Missing even one = DQ |
| 3. Env vars in `inference.py` | Checks `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` are referenced | Judges inject these vars when running your script — if your code ignores them, score = 0 |
| 4. Log markers | Checks `[START]`, `[STEP]`, `[END]` exist in `inference.py` | Judges parse stdout to extract your score — wrong format = score recorded as 0 |
| 5. Docker build | Runs `docker build` on your repo | If it doesn't build, it doesn't deploy = DQ |
| 6. openenv validate | Runs the OpenEnv framework validator | Checks your `openenv.yaml` matches the spec |

### How to run it

```bash
# macOS / Linux
chmod +x validate-submission.sh
./validate-submission.sh https://your-team.hf.space

# Windows (run inside Git Bash, not PowerShell)
bash validate-submission.sh https://your-team.hf.space
>>>>>>> kezia
```

All checks must say `PASSED` before you submit. Run this every time before pushing a final version.

---

## ⚠️ Troubleshooting — `.env` Not Loading

If you see `error=Error code: 401` and `your_api*****here` in the output, your `.env` file is not being read correctly. Here's how to diagnose:

**Check what Python is actually loading:**

```powershell
# Windows PowerShell
Get-Content .env
```

```bash
# macOS / Linux
cat .env
```

If it still shows `your_api_key_here`, your file edit didn't save. Fix and save again.

**If the file looks correct but it's still not working, set variables directly in the terminal:**

```powershell
# Windows PowerShell — run these, then immediately run inference.py WITHOUT closing the terminal
$env:HF_TOKEN = "hf_yourrealtokenhere"
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
python inference.py
```

```bash
# macOS / Linux
export HF_TOKEN="hf_yourrealtokenhere"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
python inference.py
```

**To verify what token Python loaded, add this temporarily to `inference.py` after `load_dotenv()`:**

```python
import os
print("TOKEN:", os.getenv("HF_TOKEN"))
print("URL:", os.getenv("API_BASE_URL"))
```

Run once, check the output, then remove those lines.

### Getting your real HF token

1. Go to [huggingface.co](https://huggingface.co) and log in
2. Click your profile picture → **Settings** → **Access Tokens**
3. Click **New Token** → name it anything → Role: **Read** → **Create**
4. Copy the token — it starts with `hf_`
5. Paste it into `.env` as `HF_TOKEN=hf_yourtoken`

> ⚠️ Never commit your `.env` file. Make sure `.env` is listed in `.gitignore`.

---

## ✅ Pre-Submission Checklist

Run the validator first:

```bash
./validate-submission.sh https://your-team.hf.space
```

Then verify manually:

- [ ] Real `hf_` token in `.env` (not the placeholder)
- [ ] `API_BASE_URL=https://router.huggingface.co/v1` in `.env`
- [ ] Server starts without errors: `uvicorn app:app --port 7860`
- [ ] `curl http://localhost:7860/health` returns `{"status": "ok"}`
- [ ] `python inference.py` completes and prints `[END]` line with `score=`
- [ ] No `401` errors in the `[STEP]` lines
- [ ] All three tasks run cleanly (`TASK_ID=1`, `2`, `3`)
- [ ] All scores are in `[0.0, 1.0]` range
- [ ] `.env` is in `.gitignore` and does NOT appear in `git status`
- [ ] `docker build .` succeeds locally

---



## 🐙 Git Workflow

```bash
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
```

> ⚠️ Always check `git status` before committing. `.env` must NEVER appear in the list.

---

## 🧪 stdout Log Format (mandatory per spec)

```
[START] task=basic-prioritization env=email-triage-rl model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=RESPOND reward=3.75 done=false error=null
[STEP] step=2 action=IGNORE reward=-4.20 done=false error=null
...
[END] success=true steps=20 score=0.743 rewards=3.75,-4.20,...
```

Any deviation from this format will cause incorrect evaluator scoring.
