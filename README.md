# 📧 Email Triage RL Environment

### Team Titans — Meta AI Hackathon Submission

**Live Environment:** https://TeamTitans25-Meta_ai_TeamTitans.hf.space  
**Space Repository:** https://huggingface.co/spaces/TeamTitans25/Meta_ai_TeamTitans

---

## 🧠 What Problem Are We Solving?

Most email management tools rely on **static priority labels** — simple scores that never change. But real-world inboxes are **dynamic, sequential decision-making environments** where every action has downstream consequences.

Current systems miss two critical blind spots:

**⚡ Action Cost Asymmetry**  
Not every email costs the same time. A quick "Got it" takes 2 minutes; reviewing an 8-vendor contract audit takes 3 hours.  

An intelligent agent must ask:  
*"Is the expected reward worth the actual time cost right now?"*

**👥 Sender Relationship Decay**  
Ignoring someone is not a one-time cost — it **compounds**. Ignore your boss once: tolerable. Three times: they're angry and sending escalating follow-ups.  

The agent must learn that relationship damage is an investment, not a fixed fee.

---

## 🏗️ Architecture (The Restaurant Analogy)

```
OpenEnv = A Restaurant
  server/environment.py ← The Kitchen (All game rules live here)
  server/app.py         ← The Pass (FastAPI wrapper, no logic)
  client.py             ← The Waiter (Moves data between AI and Kitchen)
  models.py             ← The Menu (Type-safe contracts for everything)
  inference.py          ← The Customer (The AI agent making decisions)
  grader.py             ← The Food Critic (Scores how well the AI did)
  data/email_bank.json  ← The Ingredients (60 pre-written email templates)
```

### Request Flow

```
inference.py → client.py → POST /step → server/app.py → environment.py
     ↑
     └──────────────── StepResponse (obs, reward, done, info) ─────────┘
```

---

## 📁 File Structure

```
meta_ai_TeamTitans/
│
├── inference.py          ← LLM agent (OpenAI client, mandatory per spec)
├── client.py             ← HTTP client connecting agent ↔ server
├── models.py             ← All data types (Email, State, Observation, Action)
├── grader.py             ← Scoring for Tasks 1, 2, 3
├── requirements.txt      ← Python dependencies
├── openenv.yaml          ← OpenEnv framework manifest
├── Dockerfile            ← Container (Python 3.11, port 7860)
├── README.md             ← This file
├── validate-submission.sh ← Pre-submission validation script
│
├── server/
│   ├── environment.py    ← Core RL logic: reset(), step(), state()
│   └── app.py            ← FastAPI endpoints: /reset, /step, /state, /health
│
├── data/
│   └── email_bank.json   ← 82 email templates 
│
├── tasks/
│   ├── task_1_easy.json  ← 20 emails, 480 min budget
│   ├── task_2_medium.json← 25 emails, 420 min budget, VIP focus
│   └── task_3_hard.json  ← 30 emails, 360 min budget, full pressure
│
└── tests/
    ├── test_environment.py ← Unit tests for environment logic
    ├── test_client.py      ← Integration tests (requires running server)
    └── test_grader.py      ← Unit tests for scoring functions
```

---

## 🐍 Setup — Python 3.11 with venv

> **Required Python version: 3.11**  
> Check what you have: `python --version` or `py -0` on Windows.

### Step 1 — Create the virtual environment
```bash
# macOS / Linux
python3.11 -m venv venv

# Windows (PowerShell)
py -3.11 -m venv venv
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

> ⚠️ **Windows permissions error?** Run this once:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Set environment variables

Create a `.env` file in the project root (never commit this file):

```env
HF_TOKEN=your_huggingface_token_here
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
TASK_ID=1
ENV_SERVER_URL=http://localhost:7860
```

### For Hugging Face Space Deployment
Set these as **Variables and Secrets** in your Space Settings:

| Name           | Value                                      | Type    |
|----------------|--------------------------------------------|---------|
| `HF_TOKEN`     | `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`        | Secret  |
| `API_BASE_URL` | `https://router.huggingface.co/v1`         | Variable|
| `MODEL_NAME`   | `Qwen/Qwen2.5-72B-Instruct`                | Variable|
| `TASK_ID`      | `1` (change for Task 2 or 3)               | Variable|

---

## 🚀 Running the Project

### Option A — Run locally (recommended for development)

**Terminal 1: Start the server**
```bash
source venv/bin/activate
cd server/
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

**Terminal 2: Run the agent**
```bash
source venv/bin/activate
python inference.py
```

To run specific tasks:
```bash
TASK_ID=1 python inference.py
TASK_ID=2 python inference.py
TASK_ID=3 python inference.py
```

### Option B — Run with Docker
```bash
docker build -t email-triage-env .
docker run -p 7860:7860 --env-file .env email-triage-env
```

### Test the API manually
```bash
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"action": 1}'
```

---

## 🎮 Environment Details

### Observation (what the agent sees per step)

| Field                  | Type   | Description                                      |
|------------------------|--------|--------------------------------------------------|
| `email_id`             | int    | Unique identifier                                |
| `sender`               | str    | Email address                                    |
| `subject`              | str    | Subject line                                     |
| `body`                 | str    | Full body text                                   |
| `sender_importance`    | str    | VIP / Normal / Spam                              |
| `email_length`         | int    | Character count — proxy for time cost            |
| `relationship_score`   | float  | Health with this sender (0–100)                  |
| `time_budget_remaining`| int    | Minutes remaining in workday                     |
| `emails_remaining`     | int    | How many emails left to process                  |

> **Partial Observability:** The agent sees one email at a time. It does **not** see actual response time, other senders’ relationship scores, or future emails.

### Action Space

| Action  | Value | Effect                                              |
|---------|-------|-----------------------------------------------------|
| IGNORE  | `0`   | Time budget unchanged. Relationship health decreases. VIPs become angry. |
| RESPOND | `1`   | Time budget decreases. Relationship health increases. |

### Reward Function

**RESPOND:**
```python
email_value = base_priority × urgency_multiplier
normalized_cost = estimated_response_time / 120.0
reward = (email_value - 5.0 × normalized_cost) × (relationship_health / 100)
```

**IGNORE:**
```python
# Spam: reward = 0
# Others:
health_penalty = 15 × importance_weight
reward = -1 × (email_value × health_penalty / 100)
```

**Episode End — Sunset Penalty:**
```python
penalty = -Σ(base_priority × relationship_health / 100) for all remaining emails
```

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
- **Goal:** Balance time efficiency, priority accuracy, and relationship health under pressure
- **Score:** `0.3 × time_efficiency + 0.4 × relationship_health + 0.3 × priority_accuracy`

All grader scores are normalized to **[0.0, 1.0]**.

---

## 🧪 stdout Log Format (Mandatory)

Your `inference.py` must emit **exactly** these formats:

```
[START] task=basic-prioritization env=email-triage-rl model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=RESPOND reward=3.75 done=false error=null
[STEP] step=2 action=IGNORE reward=-4.20 done=false error=null
...
[END] success=true steps=20 score=0.74 rewards=3.75,-4.20,...
```

**Rules:**
- `reward` values → **2 decimal places**
- `done` and `success` → lowercase (`true` / `false`)
- `error` → `null` when no error

---

## 🛡️ Pre-Submission Checklist

- [ ] Run `./validate-submission.sh https://TeamTitans25-Meta_ai_TeamTitans.hf.space`
- [ ] Real `hf_` token set in HF Space Secrets
- [ ] Server starts without errors
- [ ] All three tasks run cleanly (`TASK_ID=1,2,3`)
- [ ] All scores are in `[0.0, 1.0]`
- [ ] `.env` is in `.gitignore`
- [ ] `docker build .` succeeds locally and on HF Space

---

## ✅ Submission Checklist

- [ ] HF Space is **public** and accessible (`/reset` returns 200)
- [ ] `./validate-submission.sh` passes **all** checks
- [ ] Logs follow the exact `[START] / [STEP] / [END]` format
- [ ] All scores are between **0.0 and 1.0**
- [ ] Docker builds successfully on Hugging Face Spaces

**Submission URL:**  
https://huggingface.co/spaces/TeamTitans25/Meta_ai_TeamTitans

---

Built with ❤️ by **Team Titans** for the Meta AI Hackathon.

```

