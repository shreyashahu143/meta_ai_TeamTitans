---
title: Email Triage RL Environment
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
port : 7860
---

<div align="center">

# 📧 Email Triage RL Environment

### 🏆 Team Titans — Meta AI OpenEnv Hackathon 2026

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-HuggingFace%20Space-blue?style=for-the-badge)](https://huggingface.co/spaces/TeamTitans25/meta_ai_TeamTitans)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Server-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Container-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

**Live Environment:** https://huggingface.co/spaces/TeamTitans25/meta_ai_TeamTitans

</div>

---

## 🧠 The Problem We're Solving

Most email management tools rely on **static priority labels** — simple scores that never change. But real-world inboxes are **dynamic, sequential decision-making environments** where every action has downstream consequences.

We identified two critical blind spots that existing systems miss:

### ⚡ Blind Spot 1 — Action Cost Asymmetry
Not every email costs the same time. A quick "Got it" takes 2 minutes; reviewing an 8-vendor contract audit takes 3 hours. An intelligent agent must ask:
> *"Is the expected reward worth the actual time cost right now?"*

### 👥 Blind Spot 2 — Sender Relationship Decay
Ignoring someone is not a one-time cost — it **compounds**. Ignore your boss once: tolerable. Three times in a row: they're angry and sending escalating follow-ups.
> *The agent must learn that relationship damage is an investment, not a fixed fee.*

---

## 🏗️ Architecture

We use a **restaurant analogy** to explain the layered design:

```
OpenEnv = A Restaurant
┌─────────────────────────────────────────────────────────────────┐
│  server/environment.py  ←  The Kitchen    (All game rules)      │
│  server/app.py          ←  The Pass       (FastAPI, no logic)   │
│  client.py              ←  The Waiter     (AI ↔ Kitchen relay)  │
│  models.py              ←  The Menu       (Type-safe contracts)  │
│  inference.py           ←  The Customer   (AI agent decisions)  │
│  grader.py              ←  The Food Critic(Scores performance)  │
│  data/email_bank.json   ←  The Ingredients(82 email templates)  │
└─────────────────────────────────────────────────────────────────┘
```

### Request Flow

```
inference.py ──→ client.py ──→ POST /step ──→ server/app.py ──→ environment.py
     ↑                                                                  │
     └────────────── StepResponse (obs, reward, done, info) ───────────┘
```

---

## 📁 Project Structure

```
meta_ai_TeamTitans/
│
├── inference.py              ← LLM agent (OpenAI client, mandatory per spec)
├── client.py                 ← HTTP client connecting agent ↔ server
├── models.py                 ← All data types (Email, State, Observation, Action)
├── grader.py                 ← Scoring logic for Tasks 1, 2, and 3
├── requirements.txt          ← Python dependencies
├── openenv.yaml              ← OpenEnv framework manifest
├── Dockerfile                ← Container config (Python 3.11, port 7860)
├── README.md                 ← This file
├── validate-submission.sh    ← Pre-submission validation script
│
├── server/
│   ├── environment.py        ← Core RL logic: reset(), step(), state()
│   └── app.py                ← FastAPI endpoints: /reset /step /state /health
│
├── data/
│   └── email_bank.json       ← 82 pre-written email templates
│
├── tasks/
│   ├── task_1_easy.json      ← 20 emails · 480 min budget
│   ├── task_2_medium.json    ← 25 emails · 420 min budget · VIP focus
│   └── task_3_hard.json      ← 30 emails · 360 min budget · full pressure
│
└── tests/
    ├── test_environment.py   ← Unit tests for environment logic
    ├── test_client.py        ← Integration tests (requires running server)
    └── test_grader.py        ← Unit tests for scoring functions
```

---

## 🐍 Setup — Python 3.11

> **Required:** Python 3.11  
> Verify: `python --version` (macOS/Linux) or `py -0` (Windows)

### Step 1 — Create virtual environment

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

> ⚠️ **Windows PowerShell permissions error?** Run once:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Configure environment variables

Create a `.env` file in the project root (**never commit this file**):

```env
HF_TOKEN=your_huggingface_token_here
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
TASK_ID=1
ENV_SERVER_URL=http://localhost:7860
```

### Hugging Face Space Deployment

Set these under **Settings → Variables and Secrets** in your Space:

| Name | Value | Type |
|---|---|---|
| `HF_TOKEN` | `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx` | 🔒 Secret |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | Variable |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Variable |
| `TASK_ID` | `1` (change for Task 2 or 3) | Variable |

---

## 🚀 Running the Project

### Option A — Local Development (Recommended)

**Terminal 1 — Start the server:**
```bash
source venv/bin/activate
cd server/
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

**Terminal 2 — Run the agent:**
```bash
source venv/bin/activate
python inference.py
```

**Run specific tasks:**
```bash
TASK_ID=1 python inference.py   # Easy
TASK_ID=2 python inference.py   # Medium
TASK_ID=3 python inference.py   # Hard
```

### Option B — Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 --env-file .env email-triage-env
```

### Manual API Testing

```bash
# Health check
curl.exe https://TeamTitans25-meta-ai-teamtitans.hf.space/health

# Reset the environment
curl.exe -X POST https://TeamTitans25-meta-ai-teamtitans.hf.space/reset

# Take a step (respond to current email)
curl.exe -X POST https://TeamTitans25-meta-ai-teamtitans.hf.space/step \
  -H "Content-Type: application/json" \
  -d "{\"action\": 1}"
```

---

## 🎮 Environment Details

### Observation Space

The agent receives one email at a time with these fields:

| Field | Type | Description |
|---|---|---|
| `email_id` | `int` | Unique identifier |
| `sender` | `str` | Sender email address |
| `subject` | `str` | Email subject line |
| `body` | `str` | Full body text |
| `sender_importance` | `str` | `VIP` / `Normal` / `Spam` |
| `email_length` | `int` | Character count (proxy for time cost) |
| `relationship_score` | `float` | Sender relationship health (0–100) |
| `time_budget_remaining` | `int` | Minutes remaining in the workday |
| `emails_remaining` | `int` | Emails left to process |

> **Partial Observability:** The agent sees one email at a time. It does **not** see actual response time, other senders' relationship scores, or future emails. It must infer time cost from `email_length`.

### Action Space

| Action | Value | Effect |
|---|---|---|
| `IGNORE` | `0` | Small reading cost deducted. Relationship health decreases. VIPs become angry and send follow-ups. |
| `RESPOND` | `1` | Time budget decreases proportional to email length. Relationship health increases. |

---

## 📐 Reward Function

### On RESPOND:
```python
email_value     = base_priority × urgency_multiplier
normalized_cost = estimated_response_time / 120.0
reward          = (email_value - 5.0 × normalized_cost) × (relationship_health / 100)
```

### On IGNORE:
```python
# Spam → reward = 0  (correct decision, near-zero time cost)

# All others:
health_penalty = 15 × importance_weight
reward         = -1 × (email_value × health_penalty / 100)
# Also deducts small reading cost (20% of response time, min 2 min)
```

### Episode End — Sunset Penalty (time runs out):
```python
penalty = -Σ(base_priority × relationship_health / 100)  # for all remaining emails
```

### Episode End — Time Bonus (inbox cleared early):
```python
bonus = (time_remaining / original_time_budget) × 10
```

---

## 📊 Tasks

### Task 1 — Easy: Basic Prioritization
| Property | Value |
|---|---|
| Emails | 20 (5 VIP · 10 Normal · 5 Spam) |
| Time Budget | 480 min (generous — can recover from minor mistakes) |
| Features | Action cost asymmetry · no follow-ups · no sunset penalty |
| Goal | Respond to high-value emails, ignore spam |
| Scoring | `0.4 × value_efficiency + 0.6 × relationship_health` |

---

### Task 2 — Medium: VIP Relationship Tracking
| Property | Value |
|---|---|
| Emails | 25 (8 VIP · 12 Normal · 5 Spam) |
| Time Budget | 420 min (tighter — must skip some Normal/Spam) |
| Features | Dynamic follow-up injection · sunset penalty active |
| Goal | Learn that ignoring VIPs triggers escalating follow-up chains |
| Scoring | `0.5 × priority_accuracy + 0.5 × vip_handling_score` |
| VIP Handling Score | 50% response rate + 50% avg VIP health − 0.05 per follow-up (max −0.30) |

---

### Task 3 — Hard: Full Multi-Objective Optimization
| Property | Value |
|---|---|
| Emails | 30 (8 VIP · 14 Normal · 8 Spam) |
| Time Budget | 360 min (severe pressure — cannot finish all emails) |
| Features | Time-trap emails (up to 180 min) · dynamic follow-ups · repeat-ignore penalty · sunset penalty |
| Goal | Balance time efficiency, priority accuracy, and relationship health. Even VIP emails may not be worth responding to if they are very long and time is critically low. |
| Scoring | `0.3 × time_efficiency + 0.4 × relationship_health + 0.3 × priority_accuracy` |

> All grader scores are normalized to **[0.0, 1.0]**.

---

## 📈 Baseline Scores

> Model: `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Router

| Task | Name | Score | Notes |
|---|---|---|---|
| 1 | Basic Prioritization | **~0.94** | Correctly handles VIPs and spam; occasional suboptimal Normal choices |
| 2 | VIP Relationship Tracking | **~0.85** | Good VIP coverage; some follow-up chains triggered |
| 3 | Full Relationship Management | **~0.61** | Time pressure causes some long emails to be mishandled |

> Scores vary ±0.05 across runs due to random email ordering per episode. This variance is intentional and ensures Phase 2 grader variance checks pass.

---

## 🖨️ stdout Log Format (Mandatory)

`inference.py` must emit **exactly** these formats per the OpenEnv spec:

```
[START] task=basic-prioritization env=email-triage-rl model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=RESPOND reward=3.75 done=false error=null
[STEP] step=2 action=IGNORE reward=-4.20 done=false error=null
...
[END] success=true steps=20 score=0.720 rewards=3.75,-4.20,...
```

**Format Rules:**
- `reward` → **2 decimal places**
- `score` → **3 decimal places**
- `done` / `success` → lowercase `true` / `false`
- `error` → `null` when no error occurs
- `[END]` is **always** emitted even on exception (guaranteed by `try-finally`)

---

## 🔑 Key Design Decisions

**Why partial observability?**
The agent sees `email_length` (character count) as a proxy for time cost — not the actual `estimated_response_time`. The agent must *learn* the correlation between email length and cost, not compute it exactly. This is Blind Spot #1 made real.

**Why does ignoring cost time?**
Without a reading cost for IGNORE, ignoring is zero-cost and the environment becomes trivially exploitable (always ignore). The 20% reading cost makes the action-cost asymmetry genuine.

**Why does the grader use rewards, not ground truth?**
For Normal emails, the reward signal determines correctness rather than a hardcoded label. A positive reward means the decision was good given the time/relationship tradeoff — more faithful to the actual optimization objective.

**Why seed once at construction, not at reset?**
Re-seeding on every `reset()` call would produce identical episodes, which would fail Phase 2 variance checks. The seed is consumed only for the first episode; subsequent episodes advance the RNG naturally.

**Why task feature flags?**
The environment reads `features` from each task's JSON config to enable/disable dynamic follow-ups, sunset penalty, time traps, and repeat-ignore penalty. This ensures Task 1 (easy) has no follow-ups, Task 2 adds them, and Task 3 adds time traps and the repeat-ignore penalty.

---

## 🛡️ Pre-Submission Checklist

- [ ] Run `./validate-submission.sh https://TeamTitans25-meta-ai-teamtitans.hf.space`
- [ ] Real `hf_` token set in HF Space Secrets
- [ ] Server starts without errors
- [ ] All three tasks run cleanly (`TASK_ID=1`, `2`, `3`)
- [ ] All scores are within `[0.0, 1.0]`
- [ ] `.env` is in `.gitignore`
- [ ] `docker build .` succeeds locally

---

## ✅ Final Submission Checklist

- [ ] HF Space is **public** and accessible (`/reset` returns `200`)
- [ ] `./validate-submission.sh` passes **all 5** checks
- [ ] Logs follow the exact `[START] / [STEP] / [END]` format
- [ ] All scores are between `0.0` and `1.0`
- [ ] `score` in `[END]` uses **3 decimal places**
- [ ] Docker builds successfully on Hugging Face Spaces
- [ ] `openenv validate` passes
- [ ] `inference.py` uses **OpenAI client** (NOT Anthropic SDK)

**Submission URL:**
https://huggingface.co/spaces/TeamTitans25/Meta_ai_TeamTitans

---

<div align="center">

Built with ❤️ by **Team Titans** for the **Meta AI OpenEnv Hackathon 2026**

</div>
