# 📧 Email Triage RL Environment

> **Meta AI Hackathon — Team Titans**  
> An OpenEnv-compatible reinforcement learning environment for email inbox management.

---

Overview
The Email Triage RL Environment frames email inbox management as a sequential decision-making problem. Rather than applying static priority labels, an RL agent must learn to make optimal triage decisions under real-world constraints: limited time, dynamic sender relationships, and compounding follow-up pressure.
This environment is fully compliant with the OpenEnv specification and is designed to challenge frontier language models such as Qwen/Qwen2.5-72B-Instruct.

Problem Statement
Most email management tools assign static priority scores that never adapt. They ignore two critical dynamics of real-world inboxes:
Blind Spot 1 — Action Cost Asymmetry
Not every email costs the same effort. A quick acknowledgement takes 5 minutes; reviewing a 50-page audit report can take 3 hours. An intelligent agent must learn to ask:

"Is the expected reward of handling this email worth its true time cost right now?"

Blind Spot 2 — Sender Relationship Decay
The cost of ignoring someone is not fixed — it compounds. Repeated inaction on messages from a manager or a key client:

Degrades the relationship progressively (health: 75 → 55 → 35 → 0)
Injects escalating follow-up emails directly into the live inbox
Reduces the value of all future interactions with that sender


Environment Design
Partial Observability — The Learnable Pattern
To meet the OpenEnv rubric for Real-World Utility and Environment Design, we avoid hardcoded, unlearnable traps and instead implement a mathematically grounded partial observability mechanic.
In a real inbox, you cannot know exactly how long an email will take to handle until you begin. However, you can estimate the burden from the email's length. We model this explicitly to test an agent's ability to learn from proxy signals.
Two distinct variables govern cost:
VariableVisible to Agent?Descriptionemail_lengthYesCharacter count of the email body — the observable proxyestimated_response_timeNoTrue cost in minutes — hidden from the agent, computed deterministically from email_length (see table below)
If estimated_response_time were set independently, it would become random noise — an agent could see a 280-character email cost 180 minutes one episode and 5 the next. RL agents cannot learn from random noise.
Instead, estimated_response_time is dynamically computed from email_length using a deterministic bucketed function:
Email Length (chars)Estimated Response Time (minutes)< 1505 – 15 (quick reply)150 – 30015 – 35 (standard)300 – 50035 – 70 (deep work)> 50070 – 120 (black hole)
This creates a true learnable pattern. Over episodes, the agent learns: "A high email_length drains my time budget significantly." In Task 2 (pure time optimization), a naive "always respond" policy will hit a 500-character email, burn 90 minutes, and fail immediately. Succeeding requires learning to strategically ignore long emails under time pressure.

Architecture
Think of the system as a restaurant:
OpenEnv = A Restaurant

server/environment.py  ←  The Kitchen      (all game logic)
server/app.py          ←  The Pass         (HTTP wrapper)
client.py              ←  The Waiter       (agent ↔ server bridge)
models.py              ←  The Menu         (type-safe contracts)
inference.py           ←  The Customer     (LLM decision maker)
grader.py              ←  The Food Critic  (scores the episode)
train_agents.py        ←  The Trainer      (3 agent baselines)
Request Flow
inference.py  →  client.py  →  POST /step  →  server/app.py  →  environment.py
      ↑                                                                  ↓
      └──────────────  StepResponse(obs, reward, done, info)  ───────────┘

Project Structure
meta_ai_TeamTitans/
│
├── inference.py            ← LLM agent (OpenEnv compliant, OpenAI client)
├── grader.py               ← Scoring logic for all 3 tasks
├── client.py               ← HTTP client (reset_env, step_env, get_state)
├── models.py               ← Pydantic data contracts
├── train_agents.py         ← Trains SMART / MEDIUM / DUMB baselines
├── test_agent.py           ← Quick rule-based smoke test
├── test_Server.py          ← API endpoint tests
├── validate-submission.sh  ← Pre-submission validator (Bash script)
├── requirements.txt
├── Dockerfile
├── openenv.yaml
├── README.md
│
├── server/
│   ├── environment.py      ← Core RL logic (reset, step, state)
│   ├── app.py              ← FastAPI server (port 7860)
│   └── __init__.py
│
├── data/
│   └── email_bank.json     ← 60 email templates (18 VIP, 32 Normal, 10 Spam)
│
└── tasks/
    ├── task_1_easy.json
    ├── task_2_medium.json
    └── task_3_hard.json

Observation & Action Space
Observation Space
The agent receives one email at a time with the following fields:
FieldTypeDescriptionemail_idintUnique email identifiersenderstrSender email addresssubjectstrEmail subject linebodystrFull email bodysender_importancestrVIP / Normal / Spamemail_lengthintCharacter count — proxy for time cost (agent never sees actual minutes)relationship_scorefloatCurrent health with this sender (0–100)time_budget_remainingintMinutes remaining in the workdayemails_remainingintNumber of emails left in the inbox

Partial Observability: The agent sees one email at a time. It does not see actual time costs, other senders' relationship scores, future emails, or relationship degradation rates.

Action Space
ActionValueEffectIGNORE0Saves time. Relationship health decreases. If VIP: triggers follow-up injection.RESPOND1Costs time proportional to email length. Relationship health increases (capped at 100).
The environment uses a discrete action space with 2 actions (Discrete(2) in Gym), as defined in openenv.yaml.

Reward Function
RESPOND (action = 1)
email_value      = base_priority × urgency_multiplier
normalized_cost  = estimated_response_time / 120.0   # hidden from agent
reward           = (email_value − 5.0 × normalized_cost) × (relationship_health / 100)
IGNORE (action = 0)
if Spam:   reward = 0                                          # correct decision, no penalty
else:      health_penalty = 15 × sender_importance_weight
           reward = −1 × (email_value × health_penalty / 100)
Episode End — Sunset Penalty (time ran out, emails remain)
penalty = −Σ(base_priority × relationship_health / 100)  for all remaining emails
Episode End — Time Bonus (cleared entire inbox before time ran out)
bonus = (time_remaining / original_time_budget) × 10
Relationship Decay Table
Sender TypeIgnore PenaltyRespond BoostImportance WeightVIP−20 health+15 health3Normal−10 health+10 health2Spam000

Repeat ignore penalty: Ignoring the same sender more than once deducts an additional −10 health.

Dynamic Follow-up Injection
When a VIP is ignored, a follow-up email is dynamically inserted into the live inbox 1–2 positions ahead:
Ignore CountSubject PrefixPriority BoostExtra Health Penalty1stFOLLOW UP: {original subject}+2—2ndURGENT — SECOND FOLLOW UP: {original subject}+4 (capped at 10)−10
This makes Task 2 genuinely difficult: a cascade of VIP follow-ups can rapidly overwhelm the inbox.

Tasks
Task 1 — Easy: Basic Prioritization
ConfigValueEmails20 (5 VIP, 10 Normal, 5 Spam)Time Budget480 min (8 hours — generous)Avg time/email24 minTime PressureLow — agent can finish all emails
Scoring: 0.4 × value_efficiency + 0.6 × relationship_health
What the agent must learn:

Respond to VIP emails — ignoring hurts relationship health
Ignore Spam — responding wastes time with zero reward
Short, high-priority Normal emails are worth responding to
Very long, low-priority Normal emails may not be worth the time cost


Task 2 — Medium: VIP Relationship Tracking
ConfigValueEmails25 (8 VIP, 12 Normal, 5 Spam)Time Budget420 min (tighter)Avg time/email16.8 minTime PressureMedium — must skip some Normal and Spam
Scoring: 0.5 × priority_accuracy + 0.5 × vip_handling_score
VIP Handling Score components:

50% — VIP response rate
50% — Average VIP relationship health at episode end
−0.05 per follow-up email in inbox (max −0.30)

What the agent must learn:

Never ignore VIPs — each ignore injects a higher-priority follow-up
Follow-ups arrive 1–2 steps later, compounding the problem
Budget time carefully: skip Normal/Spam to protect capacity for VIPs


Task 3 — Hard: Full Relationship Management
ConfigValueEmails30 (8 VIP, 14 Normal, 8 Spam)Time Budget360 min (6 hours — severe)Avg time/email12 minTime PressureSevere — cannot finish all emails
Scoring: 0.3 × time_efficiency + 0.4 × relationship_health + 0.3 × priority_accuracy
Time efficiency benchmark: Compares actual reward against a perfect greedy agent (picks highest-priority emails first until time runs out).
What the agent must learn:

Cannot respond to everything — must decide what to skip
Time-trap emails (120–180 min) from VIPs may not be worth handling if time is critically low
Ignoring a VIP twice triggers a second, more urgent follow-up + extra −10 health
Sunset penalty punishes leaving high-priority emails unfinished
Optimal policy: respond to short, high-priority emails; skip long, low-priority ones regardless of sender tier


Agent Baselines
Run python train_agents.py to train and compare the three baseline agents (no LLM API key required):
AgentStrategyTask 1Task 2Task 3SMARTVIP always, Spam never, Normal by cost/time ratio~0.78~0.82~0.71MEDIUMVIP always, Spam never, Normal at 60/40 random~0.68~0.65~0.58DUMBAlways ignore everything~0.42~0.38~0.35
The SMART > MEDIUM > DUMB gap confirms the grader is sensitive to agent behavior, satisfying the OpenEnv variance check requirement.

Getting Started
Prerequisites

Python 3.11
All dependencies: pip install -r requirements.txt
A .env file in the project root (see below)

Environment Variables (.env)
envHF_TOKEN=your_huggingface_token_here
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
TASK_ID=1
ENV_SERVER_URL=http://localhost:7860
Step 1 — Start the Server
bash# From project root
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
Step 2 — Quick Smoke Test (no API key required)
bashpython test_agent.py
Step 3 — Run the LLM Agent
bash# Task 1
python inference.py

# Task 2
TASK_ID=2 python inference.py          # macOS / Linux
set TASK_ID=2 && python inference.py   # Windows

# Task 3
TASK_ID=3 python inference.py
Step 4 — Train Baseline Agents
bashpip install matplotlib
python train_agents.py
# Produces: reward_convergence_task1.png, all_agents_all_tasks.png, training_results.json
Step 5 — Manual API Tests
bashcurl http://localhost:7860/health
curl -X POST http://localhost:7860/reset
curl -X POST http://localhost:7860/step \
     -H "Content-Type: application/json" \
     -d '{"action": 1}'
curl http://localhost:7860/state

Docker
bash# Build
docker build -t email-triage-env .

# Run with environment variables (for local testing)
docker run -p 7860:7860 --env-file .env email-triage-env

# Verify
curl http://localhost:7860/health

Note: On Hugging Face Spaces, HF_TOKEN, API_BASE_URL, and MODEL_NAME are injected as secrets — you do not need to copy .env into the container. For local testing, use --env-file .env and ensure .env is present locally but never committed to Git.


OpenEnv Spec Compliance
RequirementStatusreset() / step() / state() endpoints✅openenv.yaml present and valid (includes openenv_version)✅3+ tasks with distinct scoring formulas✅Grader returns score in [0.0, 1.0]✅inference.py uses OpenAI-compatible client✅inference.py reads API_BASE_URL, MODEL_NAME, HF_TOKEN✅[START] / [STEP] / [END] stdout logging✅Dockerfile builds and runs on port 7860✅Deep copy in state()✅

Pre-Submission Validation
The validate-submission.sh script simulates the competition's automated judging robot. If Phase 1 fails, you are disqualified immediately — no human ever reviews your project.
Run this script before every submission to catch disqualifying issues yourself.
bash# macOS / Linux
chmod +x validate-submission.sh
./validate-submission.sh https://your-team.hf.space

# Windows: Use Git Bash (not PowerShell!)
bash validate-submission.sh https://your-team.hf.space

⚠️ Important: This is a Bash script. Do not run it in PowerShell or CMD — it will fail. Use Git Bash (included with Git for Windows) or WSL.

All checks must report PASSED before you submit.
What the Validator Checks
#CheckWhy It Matters1HF Space ping (POST /reset)If your Space is down → instant disqualification2Mandatory files presentMissing inference.py, openenv.yaml, requirements.txt, or Dockerfile → DQ3Env vars referenced in inference.pyJudges inject these at runtime — if your code ignores them, score = 04Log markers in inference.pyJudges parse stdout for [START], [STEP], [END] — wrong format → score recorded as 05Docker build succeedsIf it doesn't build, it doesn't deploy → DQ6openenv validate passesChecks openenv.yaml matches the spec

Troubleshooting
.env Not Loading
If you see error=Error code: 401 or your_api*****here in output, your .env file is not being read correctly.
Step 1 — Verify file contents:
bash# macOS / Linux
cat .env

# Windows PowerShell
Get-Content .env
If it still shows your_api_key_here, the file was not saved. Edit and save again.
Step 2 — Set variables directly in the terminal (fallback):
bash# macOS / Linux
export HF_TOKEN="hf_yourrealtokenhere"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
python inference.py
powershell# Windows PowerShell — run immediately before inference.py, do not close the terminal
$env:HF_TOKEN = "hf_yourrealtokenhere"
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
python inference.py
Step 3 — Debug what Python is loading (add temporarily to inference.py):
pythonimport os
print("TOKEN:", os.getenv("HF_TOKEN"))
print("URL:",   os.getenv("API_BASE_URL"))
Run once, verify the output, then remove those lines.
Getting a Hugging Face Token

Go to huggingface.co and log in
Click your profile picture → Settings → Access Tokens
Click New Token → name it → Role: Read → Create
Copy the token (starts with hf_)
Add to .env: HF_TOKEN=hf_yourtoken


Security: Never commit your .env file. Ensure .env is listed in .gitignore.


Pre-Submission Checklist
Run the validator first:
bash./validate-submission.sh https://your-team.hf.space
Then verify the following manually:

 Real hf_ token in .env (not the placeholder)
 API_BASE_URL=https://router.huggingface.co/v1 in .env
 Server starts without errors: uvicorn server.app:app --port 7860
 curl http://localhost:7860/health returns {"status": "ok"}
 python inference.py completes and prints an [END] line with score=
 No 401 errors in any [STEP] lines
 All three tasks run cleanly (TASK_ID=1, 2, 3)
 All scores fall within [0.0, 1.0]
 .env is in .gitignore and does not appear in git status
 docker build . succeeds locally
 openenv validate passes in the repo root


Git Workflow
bash# First-time setup
git init
git add .
git commit -m "initial project structure"
git remote add origin https://github.com/YOUR_USERNAME/meta_ai_TeamTitans.git
git branch -M main
git push -u origin main

# Daily workflow
git pull                              # always pull before starting work
# ... make changes ...
git add .
git commit -m "describe what you changed"
git push

Always run git status before committing. .env must never appear in the staged file list.


Log Format
The following stdout format is mandatory per the OpenEnv spec. Any deviation will cause the evaluator to record an incorrect score.
[START] task=basic-prioritization env=email-triage-rl model=Qwen/Qwen2.5-72B-Instruct
[STEP]  step=1  action=RESPOND  reward=3.75   done=false  error=null
[STEP]  step=2  action=IGNORE   reward=-4.20  done=false  error=null
...
[END]   success=true  steps=20  score=0.743  rewards=3.75,-4.20,...

Note: The task field in [START] can be any descriptive string; the judge verifies only the presence of [START], [STEP], and [END] markers.


Built with care by Team Titans for the Meta AI Hackathon.