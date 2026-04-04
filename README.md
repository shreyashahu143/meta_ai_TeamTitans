# 📧 Email Triage RL Environment


## 🧠 What Problem Are We Solving?

Most email management tools rely on static priority labels — simple scores that never change. But real-world email inboxes are dynamic, sequential decision-making environments where every action has consequences over time.
Current systems completely miss two critical blind spots:
⚡ Action Cost Asymmetry
Not every email costs the same time and effort. A quick “Got it” reply takes 30 seconds, while reviewing a 50-page report can take 2 hours. An intelligent agent must learn to ask:
“Is the expected reward of this email worth the actual time cost right now?”
👥 Sender Relationship Decay
The cost of ignoring someone is not fixed — it compounds with every delay. Ignoring your boss or a key stakeholder once may be tolerable, but repeated inaction damages the relationship, reduces future cooperation, and shrinks the long-term value of responding.

---

## 🏗️ Project Architecture (The Restaurant Analogy)

```
OpenEnv = A Restaurant

  server/         ← The Kitchen       (Rules live here, in a Docker container)
  client.py       ← The Waiter        (Moves info between AI and Kitchen)
  models.py       ← The Menu          (Defines what can be ordered/served)
  inference.py    ← The Customer      (The AI that makes decisions)
  grader.py       ← The Food Critic   (Scores how well the AI did)
```

### The 3-Component Pattern (from OpenEnv spec)

```
email_triage_env/
├── models.py          ← Type-safe contracts (Action, Observation, State)
├── client.py          ← HTTPEnvClient implementation (what YOU import in inference.py)
└── server/
    ├── environment.py ← Game/simulation logic
    ├── app.py         ← FastAPI server wrapper
    └── Dockerfile     ← Container definition
```

### How the Client Talks to the Server

```
AI (inference.py)
     │
     │  calls
     ▼
client.py (inherits HTTPEnvClient)
     │  implements 3 methods:
     │  1. _step_payload()   → converts Action → JSON
     │  2. _parse_result()   → converts JSON → Observation
     │  3. _parse_state()    → converts JSON → State
     │
     │  sends HTTP POST/GET
     ▼
server/app.py  (FastAPI on http://localhost:8000)
     │
     │  calls
     ▼
server/environment.py  (core RL logic)
```

---

## 📁 Full File Structure

```
email_triage_env/
│
├── models.py                  ← Data contracts (Email, State, Observation, Action)
├── client.py                  ← HTTP client (connects inference.py to server)
├── inference.py               ← LLM agent that plays the game
├── grader.py                  ← Scoring logic for tasks 1, 2, 3
├── requirements.txt           ← Python dependencies
├── openenv.yaml               ← OpenEnv framework config (for HF Spaces)
├── README.md                  ← This file
│
├── server/
│   ├── environment.py         ← Core RL logic: reset(), step(), state()
│   ├── app.py                 ← FastAPI wrapper (exposes /reset, /step, /state)
│   └── Dockerfile             ← Container: Python 3.11, port 8000
│
├── data/
│   └── email_bank.json        ← Pre-written email templates (VIP/Normal/Spam)
│
├── tasks/
│   ├── task_1_easy.json       ← 20 emails, full time budget, basic scoring
│   ├── task_2_medium.json     ← VIP tracking focus, partial observability
│   └── task_3_hard.json       ← Full relationship decay, time pressure
│
└── tests/
    ├── test_environment.py    ← Unit tests for reset(), step(), reward math
    ├── test_grader.py         ← Unit tests for scoring functions
    └── test_client.py         ← Integration tests: client ↔ server roundtrip
```

---

## 🐍 Virtual Environment Setup

> **Standard for this hackathon:** Python **3.12** with `venv`. Do NOT use conda or poetry .
>
> To check which Python versions you have installed, run:
> ```powershell
> py -0
> ```
> You need **3.12** or **3.9** (3.12 preferred). If you don't have either, download from python.org.

### Step 1: Create the virtual environment

```powershell
# Windows (PowerShell) — use py launcher to pick the right version
py -3.12 -m venv venv
```

```bash
# macOS/Linux
python3.12 -m venv venv
```

### Step 2: Activate it

```powershell
# Windows PowerShell
venv\Scripts\Activate.ps1
```

> ⚠️ **If you get a permissions error on Windows**, run this once and try again:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

```bash
# macOS/Linux
source venv/bin/activate
```

```bat
:: Windows Command Prompt
venv\Scripts\activate.bat
```

### Step 3: Copy and fill in your .env file

```powershell
copy .env.example .env
```

```bash
# macOS/Linux
cp .env.example .env
```

Then open `.env` and paste in your `ANTHROPIC_API_KEY`.

### Step 4: Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 5: Verify

```powershell
python --version   # Should show Python 3.12.x
pip list           # Should show fastapi, pydantic, anthropic, etc.
```

> ⚠️ **Always activate the venv before running ANY file in this project.**
> You'll know it's active when you see `(venv)` at the start of your terminal line.

---

## 🐙 Pushing to GitHub

Do this **once** when setting up the repo for the first time.

### Step 1: Initialize git

```powershell
git init
```

### Step 2: Check what will be committed

```powershell
git status
```

> ⚠️ Make sure `.env` does **NOT** appear in the list. If it does, stop — something is wrong with `.gitignore`.

### Step 3: Stage all files

```powershell
git add .
```

### Step 4: First commit

```powershell
git commit -m "initial project structure"
```

### Step 5: Create repo on GitHub

Go to **github.com → New Repository** → name it `meta_ai_TeamTitans` → **do NOT check "add README"** → click Create → copy the repo URL.

### Step 6: Connect and push

```powershell
git remote add origin https://github.com/YOUR_USERNAME/meta_ai_TeamTitans.git
git branch -M main
git push -u origin main
```

### Teammates: clone the repo

```powershell
git clone https://github.com/YOUR_USERNAME/meta_ai_TeamTitans.git
cd meta_ai_TeamTitans
py -3.12 -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # then fill in your own API key
```

### Day-to-day workflow (everyone)

```powershell
git pull                        # always pull before starting work
# ... make your changes ...
git add .
git commit -m "what you did"
git push
```

---

## 🚀 Running the Project

### Option A: Run server locally (no Docker)

```bash
# Terminal 1 — Start the server
source venv/bin/activate
cd server/
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Run the agent
source venv/bin/activate
python inference.py --task 1
```

### Option B: Run server with Docker

```bash
# Build the container
cd server/
docker build -t email-triage-env .

# Run the container
docker run -p 8000:8000 email-triage-env

# In another terminal, run the agent
python inference.py --task 1
```

### Test the API manually

```bash
# Check server is running
curl http://localhost:8000/health

# Reset the environment
curl -X POST http://localhost:8000/reset

# Take an action (0=ignore, 1=respond)
curl -X POST http://localhost:8000/step -H "Content-Type: application/json" -d '{"action": 1}'

# Get current state
curl http://localhost:8000/state
```

---

## 🎮 Environment Details

### State Space

| Field | Type | Description |
|-------|------|-------------|
| `inbox` | `List[Email]` | All pending emails in the episode |
| `current_email_index` | `int` | Pointer to the current email |
| `relationships` | `Dict[str, Relationship]` | Health score per sender (0–100) |
| `current_time` | `int` | Current timestep |
| `time_budget_remaining` | `int` | Minutes left in the workday (starts: 480) |
| `total_time_spent` | `int` | Accumulated response time so far |

### Observation (what the agent sees per step)

| Field | Description |
|-------|-------------|
| `email_id` | Unique ID |
| `sender` | Email address |
| `subject` | Email subject line |
| `body` | Full body text |
| `sender_importance` | VIP / Normal / Spam |
| `email_length` | Character count (proxy for time cost — agent does NOT see actual cost) |
| `relationship_score` | Current health with this sender |
| `time_budget_remaining` | Minutes left |
| `emails_remaining` | How many emails are left in inbox |

> **Partial Observability:** Agent sees ONE email at a time. It does NOT see future emails, actual time costs, or other senders' relationship scores.

### Action Space

| Action | Value | Effect |
|--------|-------|--------|
| Ignore | `0` | Relationship health decreases. Time budget unchanged. |
| Respond | `1` | Relationship health increases (capped at 100). Time budget decreases by `action_cost`. |

### Reward Function

**On RESPOND (action = 1):**
```
reward = (email_value - 0.5 × action_cost) × (relationship_health / 100)
```

**On IGNORE (action = 0):**
```
health_penalty = 15 × sender_importance_weight
reward = -1 × (email_value × health_penalty / 100)
```

**Episode End — Sunset Penalty (if time runs out with emails remaining):**
```
FinalPenalty = Σ(remaining_email_values × relationship_scores)
```

### Relationship Decay Table

| Sender Type | Ignore Penalty | Respond Boost | Followup Multiplier |
|-------------|----------------|---------------|---------------------|
| VIP | −20 health | +15 health | 1.5× |
| Normal | −10 health | +10 health | 1.2× |
| Spam | 0 | 0 | 1.0× |

---

## 📊 Tasks

### Task 1 — Easy: Basic Prioritization
- **Emails:** 20 (5 VIP, 10 Normal, 5 Spam)
- **Time Budget:** 480 minutes
- **Goal:** Maximize total value while keeping all relationships healthy
- **Score:** `0.4 × (value_gained / max_value) + 0.6 × (avg_relationship_health / 100)`

### Task 2 — Medium: VIP Tracking
- **Emails:** 25 (8 VIP, 12 Normal, 5 Spam) with follow-up chains
- **Time Budget:** 420 minutes
- **Goal:** Learn that ignoring VIPs triggers costly follow-ups
- **Score:** `0.5 × correct_priority_rate + 0.5 × vip_handling_score`

### Task 3 — Hard: Full Relationship Management
- **Emails:** 30 (mixed, with angry follow-ups and time traps)
- **Time Budget:** 360 minutes
- **Goal:** Balance time, priority, AND relationship health across all senders
- **Score:** `0.3 × efficiency + 0.4 × relationship_health + 0.3 × priority_accuracy`

---

## 👥 Team Responsibilities

| File | Owner | Description |
|------|-------|-------------|
| `models.py` | System Architect | Data structures — done first, everyone depends on this |
| `server/environment.py` | Algorithm Engineer | Core RL logic: reset, step, reward |
| `server/app.py` | Algorithm Engineer | FastAPI endpoints wrapping environment |
| `server/Dockerfile` | Algorithm Engineer | Container setup |
| `client.py` | LLM Engineer | HTTP client connecting agent to server |
| `inference.py` | LLM Engineer | The AI agent and prompt engineering |
| `grader.py` | LLM Engineer | Scoring all 3 tasks |
| `data/email_bank.json` | Content Lead | Pre-written email templates |
| `tasks/*.json` | System Architect | Task config files |

---



## 🔑 Environment Variables

Create a `.env` file in the project root (never commit this):

```env
ANTHROPIC_API_KEY=your_key_here
ENV_SERVER_URL=http://localhost:8000
TASK_ID=1
```

---

## ✅ Success Criteria

### Minimum Viable Product
- [ ] `reset()` returns a valid State with 15–20 emails
- [ ] `step(0)` and `step(1)` return correct (observation, reward, done) tuples
- [ ] Relationship health degrades on ignore, improves on respond
- [ ] Task 1 runs end-to-end without errors
- [ ] Grader produces a score between 0–100

### Stretch Goals
- [ ] All 3 tasks implemented and tested
- [ ] Follow-up email escalation working (angry VIP logic)
- [ ] Live demo on Hugging Face Spaces
- [ ] Sunset penalty applied correctly when time runs out

---

