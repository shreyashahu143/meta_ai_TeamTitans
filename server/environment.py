"""
server/environment.py — Core RL Environment Logic
==================================================

PURPOSE:
    This is the "Kitchen" of our restaurant. All the game rules live here.
    No HTTP, no FastAPI — just pure Python logic.

    Implements the OpenEnv standard interface:
        reset()  → initializes a new episode, returns first Observation
        step()   → takes an action, returns (Observation, reward, done)
        state()  → returns a deep copy of the full internal State (for grader/debug)

CONNECTS TO:
    ← models.py          (imports Email, State, Relationship, EmailObservation, etc.)
    ← data/email_bank.json (loads pre-written email templates)
    → server/app.py      (app.py calls reset/step/state and serializes output to JSON)

OWNER: Algorithm Engineer
"""

import copy
import json
import os
import random
from typing import Dict, List, Optional, Tuple

from models import (
    RELATIONSHIP_CONFIG,
    Action,
    Email,
    EmailObservation,
    Relationship,
    State,
    StepResponse,
)


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

DEFAULT_TIME_BUDGET = 480       # 8-hour workday in minutes
INITIAL_RELATIONSHIP_HEALTH = 75.0  # All senders start at 75/100 health

# Time cost distribution:
#   60% Quick      (5-15 min)  — "Thanks", "Acknowledged"
#   30% Deep Work  (30-60 min) — Reports, meeting prep
#   10% Black Hole (120 min)   — Time trap, even from VIPs
TIME_COST_DISTRIBUTION = {
    "quick":      {"range": (5, 15),    "weight": 0.6},
    "deep":       {"range": (30, 60),   "weight": 0.3},
    "black_hole": {"range": (120, 120), "weight": 0.1},
}


class EmailTriageEnv:
    """
    The Email Triage RL Environment.

    One "episode" = one simulated workday.
    Each "step" = the agent decides to IGNORE (0) or RESPOND (1) to one email.

    Episode ends when:
        1. Inbox is empty (current_email_index >= len(inbox))  ← clean finish
        2. Time budget runs out (time_budget_remaining <= 0)   ← forced end

    If time runs out with emails remaining → apply Sunset Penalty:
        FinalPenalty = Σ(remaining_email_values × relationship_scores / 100)
    """

    def __init__(self, task_config: Optional[dict] = None):
        """
        Args:
            task_config: Dict loaded from tasks/task_1_easy.json etc.
                         Controls num_emails, distribution, time_budget.
                         If None, uses Task 1 defaults.
        """
        self.task_config = task_config or {
            "num_emails": 20,
            "vip_count": 5,
            "normal_count": 10,
            "spam_count": 5,
            "time_budget": DEFAULT_TIME_BUDGET,
        }

        # Internal state (private — only exposed via state() deep copy)
        self._inbox: List[Email] = []
        self._relationships: Dict[str, Relationship] = {}
        self._current_email_index: int = 0
        self._current_timestep: int = 0
        self._time_budget_remaining: int = DEFAULT_TIME_BUDGET
        self._total_time_spent: int = 0
        self._emails_handled: int = 0
        self._ignored_senders: Dict[str, int] = {}   # sender_email → ignore count

        # Load email templates
        self._email_bank = self._load_email_bank()

    # -----------------------------------------------------------------------
    # PUBLIC INTERFACE (called by app.py)
    # -----------------------------------------------------------------------

    def reset(self) -> EmailObservation:
        """
        Start a new episode.

        Steps:
            1. Generate emails from email_bank (VIP/Normal/Spam mix)
            2. Shuffle inbox order (randomize difficulty)
            3. Initialize all relationship healths at 75
            4. Reset all counters
            5. Return first email as Observation

        Returns:
            EmailObservation: The first email the agent should evaluate.
        """
        cfg = self.task_config
        vip_emails    = random.sample(self._email_bank.get("vip_emails", []),
                                      min(cfg["vip_count"], len(self._email_bank.get("vip_emails", []))))
        normal_emails = random.sample(self._email_bank.get("normal_emails", []),
                                      min(cfg["normal_count"], len(self._email_bank.get("normal_emails", []))))
        spam_emails   = random.sample(self._email_bank.get("spam_emails", []),
                                      min(cfg["spam_count"], len(self._email_bank.get("spam_emails", []))))

        all_emails = vip_emails + normal_emails + spam_emails
        random.shuffle(all_emails)

        # Build Email objects with IDs
        self._inbox = []
        for i, template in enumerate(all_emails):
            email = Email(
                email_id=i,
                sender=template["sender"],
                sender_domain=template.get("sender_domain", "external"),
                subject=template["subject"],
                body=template["body"],
                sender_importance=template["sender_importance"],
                base_priority=template["base_priority"],
                estimated_response_time=self._sample_time_cost(template),
                received_at_timestep=0,
            )
            self._inbox.append(email)

        # Initialize relationships for all unique senders
        self._relationships = {}
        for email in self._inbox:
            if email.sender not in self._relationships:
                cfg_rel = RELATIONSHIP_CONFIG[email.sender_importance]
                self._relationships[email.sender] = Relationship(
                    sender_email=email.sender,
                    health=INITIAL_RELATIONSHIP_HEALTH,
                    importance=email.sender_importance,
                    importance_weight=cfg_rel["importance_weight"],
                    degradation_rate=abs(cfg_rel["ignore_penalty"]),
                )

        # Reset counters
        self._current_email_index = 0
        self._current_timestep = 0
        self._time_budget_remaining = self.task_config.get("time_budget", DEFAULT_TIME_BUDGET)
        self._total_time_spent = 0
        self._emails_handled = 0
        self._ignored_senders = {}

        return self._build_observation()

    def step(self, action: int) -> StepResponse:
        """
        Execute one action and advance the environment.

        Args:
            action: 0 = IGNORE, 1 = RESPOND

        Returns:
            StepResponse with (observation, reward, done, info)

        REWARD LOGIC:
            RESPOND (1):
                reward = (email_value - 0.5 × action_cost) × (relationship_health / 100)
                → Relationship health += respond_boost (capped at 100)
                → time_budget_remaining -= action_cost

            IGNORE (0):
                health_penalty = 15 × sender_importance_weight
                reward = -1 × (email_value × health_penalty / 100)
                → Relationship health -= ignore_penalty
                → Time budget unchanged
                → If sender was ignored before: urgency_multiplier += 0.5, extra -10 health
                → If VIP: set is_angry = True (triggers follow-up email logic)
        """
        if self._current_email_index >= len(self._inbox):
            # Already done — shouldn't happen, but guard anyway
            return StepResponse(
                observation=self._build_observation(),
                reward=0.0,
                done=True,
                info={"error": "step() called after episode ended"},
            )

        current_email = self._inbox[self._current_email_index]
        relationship = self._relationships[current_email.sender]

        # ------ Calculate reward ------
        if action == Action.RESPOND:
            reward, time_cost = self._reward_respond(current_email, relationship)
            self._time_budget_remaining -= time_cost
            self._total_time_spent += time_cost
            relationship.health = min(100.0, relationship.health +
                                      RELATIONSHIP_CONFIG[current_email.sender_importance]["respond_boost"])
            relationship.interaction_count += 1
            info = {"action": "respond", "time_cost": time_cost, "reward": reward}

        else:  # IGNORE
            reward = self._reward_ignore(current_email, relationship)
            penalty = RELATIONSHIP_CONFIG[current_email.sender_importance]["ignore_penalty"]
            relationship.health = max(0.0, relationship.health + penalty)  # penalty is negative

            # Track ignores — escalation logic
            sender = current_email.sender
            self._ignored_senders[sender] = self._ignored_senders.get(sender, 0) + 1
            current_email.times_ignored += 1
            current_email.sender_urgency_multiplier += 0.5

            if self._ignored_senders[sender] > 1:
                relationship.health = max(0.0, relationship.health - 10)  # Extra repeat penalty

            if current_email.sender_importance == "VIP":
                relationship.is_angry = True   # Triggers follow-up injection

            info = {
                "action": "ignore",
                "ignore_count": self._ignored_senders.get(sender, 0),
                "relationship_delta": penalty,
                "reward": reward,
            }

        self._emails_handled += 1
        self._current_email_index += 1
        self._current_timestep += 1

        # ------ Check done conditions ------
        inbox_empty = self._current_email_index >= len(self._inbox)
        time_up = self._time_budget_remaining <= 0

        done = inbox_empty or time_up

        if done and time_up and not inbox_empty:
            # Apply Sunset Penalty for unhandled emails
            sunset_penalty = self._calculate_sunset_penalty()
            reward += sunset_penalty
            info["sunset_penalty"] = sunset_penalty

        next_obs = self._build_observation() if not done else self._build_final_observation()

        return StepResponse(
            observation=next_obs,
            reward=round(reward, 4),
            done=done,
            info=info,
        )

    def state(self) -> State:
        """
        Returns the full internal state as a DEEP COPY.

        WHY DEEP COPY: If you return references to live objects, the grader
        will record the SAME (final) state for every step in the history.
        Deep copy ensures each recorded state is a frozen snapshot.

        Returns:
            State: Complete environment state, safe to store in episode history.
        """
        return State(
            inbox=copy.deepcopy(self._inbox),
            current_email_index=self._current_email_index,
            relationships=copy.deepcopy(self._relationships),
            current_timestep=self._current_timestep,
            time_budget_remaining=self._time_budget_remaining,
            total_time_spent=self._total_time_spent,
            emails_handled=self._emails_handled,
        )

    # -----------------------------------------------------------------------
    # PRIVATE HELPERS
    # -----------------------------------------------------------------------

    def _reward_respond(self, email: Email, relationship: Relationship) -> Tuple[float, int]:
        """
        Calculates reward for RESPOND action.
        Formula: (email_value - 0.5 × action_cost) × (relationship_health / 100)
        """
        email_value = email.base_priority * email.sender_urgency_multiplier
        action_cost = email.estimated_response_time
        reward = (email_value - 0.5 * action_cost) * (relationship.health / 100)
        return round(reward, 4), action_cost

    def _reward_ignore(self, email: Email, relationship: Relationship) -> float:
        """
        Calculates reward for IGNORE action (always negative for non-spam).
        Formula: -1 × (email_value × health_penalty / 100)
        """
        if email.sender_importance == "Spam":
            return 0.0   # Correctly ignoring spam — no penalty
        health_penalty = 15 * relationship.importance_weight
        email_value = email.base_priority * email.sender_urgency_multiplier
        reward = -1 * (email_value * health_penalty / 100)
        return round(reward, 4)

    def _calculate_sunset_penalty(self) -> float:
        """
        Applied when time runs out with emails remaining.
        FinalPenalty = Σ(remaining_email_values × relationship_scores / 100)
        """
        penalty = 0.0
        for i in range(self._current_email_index, len(self._inbox)):
            email = self._inbox[i]
            rel = self._relationships.get(email.sender)
            if rel:
                penalty -= email.base_priority * (rel.health / 100)
        return round(penalty, 4)

    def _build_observation(self) -> EmailObservation:
        """Build agent's limited view of the current email."""
        if self._current_email_index >= len(self._inbox):
            return self._build_final_observation()

        email = self._inbox[self._current_email_index]
        rel = self._relationships.get(email.sender)
        rel_score = rel.health if rel else 50.0
        emails_remaining = len(self._inbox) - self._current_email_index - 1

        return EmailObservation(
            email_id=email.email_id,
            sender=email.sender,
            subject=email.subject,
            body=email.body,
            sender_importance=email.sender_importance,
            email_length=len(email.body),          # Proxy — NOT actual time cost
            relationship_score=rel_score,
            time_budget_remaining=self._time_budget_remaining,
            emails_remaining=emails_remaining,
        )

    def _build_final_observation(self) -> EmailObservation:
        """Dummy observation returned when episode is done."""
        return EmailObservation(
            email_id=-1,
            sender="done@done.com",
            subject="Episode Complete",
            body="",
            sender_importance="Spam",
            email_length=0,
            relationship_score=0.0,
            time_budget_remaining=self._time_budget_remaining,
            emails_remaining=0,
        )

    def _sample_time_cost(self, template: dict) -> int:
        """
        Assigns a time cost based on distribution:
            60% Quick (5-15 min), 30% Deep (30-60 min), 10% Black Hole (120 min)

        Note: A VIP can get a "Black Hole" email. If the agent spends 2 hours on
        a VIP's "lunch tomorrow?" email, it gets a negative reward. This ensures
        the agent reads CONTENT, not just sender name.
        """
        if "estimated_response_time" in template:
            return template["estimated_response_time"]  # Use explicit override if set

        roll = random.random()
        if roll < 0.6:
            return random.randint(5, 15)
        elif roll < 0.9:
            return random.randint(30, 60)
        else:
            return 120

    def _load_email_bank(self) -> dict:
        """Load pre-written email templates from data/email_bank.json"""
        # Try multiple paths (works whether run from root or server/)
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "..", "data", "email_bank.json"),
            os.path.join(os.path.dirname(__file__), "data", "email_bank.json"),
            "data/email_bank.json",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, "r") as f:
                    return json.load(f)

        # Fallback: minimal inline bank so env still runs during development
        return {
            "vip_emails": [
                {"sender": "boss@company.com", "sender_domain": "internal",
                 "subject": "Q4 Report Review Needed", "body": "Please review the attached Q4 report and share feedback by EOD.", "sender_importance": "VIP", "base_priority": 9},
                {"sender": "cto@company.com", "sender_domain": "internal",
                 "subject": "Architecture Decision Needed", "body": "We need a decision on the new microservices architecture. Please review the doc and respond.", "sender_importance": "VIP", "base_priority": 8},
            ],
            "normal_emails": [
                {"sender": "teammate@company.com", "sender_domain": "internal",
                 "subject": "PR Review Request", "body": "Hey, can you review my pull request when you get a chance? It's a small fix.", "sender_importance": "Normal", "base_priority": 5},
            ],
            "spam_emails": [
                {"sender": "newsletter@random.com", "sender_domain": "external",
                 "subject": "Top 10 Productivity Hacks!", "body": "Unsubscribe at any time. Limited offer today only!", "sender_importance": "Spam", "base_priority": 1},
            ],
        }
