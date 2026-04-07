# 📧 Email Triage RL Environment

> **Meta AI Hackathon — Team Titans**  
> An OpenEnv-compatible reinforcement learning environment for intelligent email inbox management.

---

## ✨ Overview

The **Email Triage RL Environment** transforms chaotic email inboxes into a rich, sequential decision-making problem for Reinforcement Learning agents.

Instead of static priority labels, the agent must learn to make optimal triage decisions while balancing **limited time**, **varying action costs**, and **decaying sender relationships** — mirroring real-world email challenges.

Fully compliant with **Meta OpenEnv** specification and built to challenge frontier LLMs like Qwen/Qwen2.5-72B-Instruct.

---

## 🧠 The Problem We’re Solving

Most email management tools rely on **static priority scores** that never change. They completely ignore two critical real-world blind spots:

**⚡ Action Cost Asymmetry**  
Not every email costs the same time and effort.  
A quick “Got it” reply takes ~5 minutes, while reviewing a 50-page report can take up to 3 hours.  

The intelligent agent must learn to ask:  
**“Is the expected reward of handling this email worth its true time cost right now?”**

**👥 Sender Relationship Decay**  
The cost of ignoring someone is not fixed — it **compounds** over time.  
Repeated inaction on messages from your boss or key stakeholders:

- Progressively degrades relationship health (e.g., 75 → 55 → 35 → 0)
- Triggers escalating follow-up emails injected directly into the inbox
- Reduces the long-term value of future interactions

---

## 🚀 Key Features

- **Partial Observability**: Agent sees `email_length` (proxy) but not the hidden actual response time
- **Learnable Cost Model**: Deterministic mapping from email length → time cost (no random noise)
- **Dynamic Relationship Tracking**: Health decays on ignore, improves on respond
- **VIP Follow-up Escalation**: Ignored VIPs generate increasingly urgent follow-ups
- **Time Budget & Sunset Penalty**: Real pressure — you cannot answer everything
- **3 Progressive Difficulty Levels**: Easy → Medium → Hard

### Observation Space (One email at a time)
- `email_id`, `sender`, `subject`, `body`
- `sender_importance` (VIP / Normal / Spam)
- `email_length` — observable proxy for cost
- `relationship_score` (0–100)
- `time_budget_remaining`, `emails_remaining`

### Action Space (Discrete)
- `0` → IGNORE  
- `1` → RESPOND

---

## ✅ Success Criteria

### Minimum Viable Product
- [ ] `reset()` returns a valid state with 15–20 emails
- [ ] `step()` returns correct `(observation, reward, terminated, truncated, info)` tuple
- [ ] Relationship health correctly degrades on ignore and improves on respond
- [ ] Task 1 runs end-to-end without errors
- [ ] Grader produces a normalized score between 0.0 and 1.0

### Stretch Goals
- [ ] All 3 tasks fully implemented and tested
- [ ] Follow-up email escalation with “angry VIP” logic
- [ ] Live demo deployed on Hugging Face Spaces
- [ ] Sunset penalty and time bonus applied correctly

---

## 📁 Project Structure

```bash
meta_ai_TeamTitans/
├── inference.py              # Main LLM agent
├── grader.py                 # Scoring logic for 3 tasks
├── client.py                 # HTTP client for environment
├── models.py                 # Pydantic data models
├── train_agents.py           # Baseline agents (SMART / MEDIUM / DUMB)
├── server/
│   ├── environment.py        # Core RL environment logic
│   └── app.py                # FastAPI server
├── data/
│   └── email_bank.json
├── tasks/
│   ├── task_1_easy.json
│   ├── task_2_medium.json
│   └── task_3_hard.json
├── openenv.yaml
├── Dockerfile
├── validate-submission.sh
└── requirements.txt

🏗️ Architecture
Think of the system as a restaurant:

server/environment.py → Kitchen (core game logic)
server/app.py         → Pass (FastAPI HTTP server)
client.py             → Waiter (bridge between agent and server)
inference.py          → Customer (LLM decision maker)
grader.py             → Food Critic (evaluator)

Request Flow:
inference.py → client.py → POST /step → server/app.py → environment.py

📊 Tasks Overview

TaskDifficultyEmailsTime BudgetMain FocusScoring Weights1Easy20480 minBasic prioritization0.4 eff + 0.6 rel2Medium25420 minVIP relationship tracking0.5 pri + 0.5 VIP3Hard30360 minFull time & relationship management0.3 eff + 0.4 rel + 0.3 pri

🧪 Agent Baselines
Run python train_agents.py to compare three baselines:

SMART — VIP-first, cost-aware → Scores: ~0.78 / 0.82 / 0.71
MEDIUM — Partial random on Normal → Scores: ~0.68 / 0.65 / 0.58
DUMB — Always ignore → Scores: ~0.42 / 0.38 / 0.35


🚀 Getting Started
Prerequisites

Python 3.11+
pip install -r requirements.txt

1. Start the Server
Bashuvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
2. Quick Smoke Test
Bashpython test_agent.py
3. Run the LLM Agent
Bash# Task 1
python inference.py

# Task 2
TASK_ID=2 python inference.py

# Task 3  
TASK_ID=3 python inference.py
Docker Support
Bashdocker build -t email-triage-env .
docker run -p 7860:7860 --env-file .env email-triage-env

📋 OpenEnv Compliance

✅ reset() / step() endpoints implemented
✅ openenv.yaml present and valid
✅ 3 distinct tasks with different scoring
✅ Proper [START], [STEP], [END] logging
✅ Dockerfile ready for Hugging Face Spaces
✅ Grader returns score in [0.0, 1.0]


🛡️ Pre-Submission Validation
Run this before submitting:
Bashchmod +x validate-submission.sh
./validate-submission.sh https://your-team.hf.space
All checks must pass to avoid disqualification.

📝 Built with ❤️ by Team Titans
For the Meta AI Hackathon — Creating realistic RL environments for real-world problems.