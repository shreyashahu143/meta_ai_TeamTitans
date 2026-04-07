"""
server/environment.py — Core RL Environment Logic
==================================================

PURPOSE:
    The "Kitchen" of our restaurant analogy. All game rules live here.
    No HTTP, no FastAPI — just pure Python logic.

    OpenEnv standard interface:
        reset()  → initialises a new episode, returns first Observation
        step()   → takes an action, returns StepResponse(Observation, reward, done, info)
        state()  → returns a DEEP COPY of full internal State (for grader/debug)

REWARD FUNCTION:
    RESPOND (1):
        email_value     = base_priority × urgency_multiplier   (range: 1–10+)
        normalized_cost = estimated_response_time / 120.0      (range: 0–1.5 for 180-min emails)
        reward = (email_value - 5.0 × normalized_cost) × (relationship_health / 100)

    IGNORE (0):
        if Spam → reward = 0 (correct decision, no penalty)
        else    → health_penalty = 15 × importance_weight
                  reward = -1 × (email_value × health_penalty / 100)

    Episode end — Sunset Penalty (if time runs out with emails remaining):
        penalty = -Σ(base_priority × relationship_health / 100) for remaining emails

    Episode end — Time Bonus (if inbox fully cleared before time runs out):
        bonus = (time_remaining / original_time_budget) × 10

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

DEFAULT_TIME_BUDGET         = 480       # 8-hour workday in minutes
INITIAL_RELATIONSHIP_HEALTH = 75.0     # All senders start at 75/100


class EmailTriageEnv:
    """
    The Email Triage RL Environment.

    One episode = one simulated workday.
    Each step = agent decides IGNORE (0) or RESPOND (1) to one email.

    Episode ends when:
        1. Inbox is empty (current_email_index >= len(inbox))  — clean finish
        2. Time budget runs out (time_budget_remaining <= 0)   — forced end
    """

    def __init__(self, task_config: Optional[dict] = None):
        """
        Args:
            task_config: Dict from tasks/task_X.json.
                         Controls num_emails, distribution, time_budget.
                         If None, uses Task 1 defaults.
        """
        self.task_config = task_config or {
            "num_emails":    20,
            "vip_count":     5,
            "normal_count":  10,
            "spam_count":    5,
            "time_budget":   DEFAULT_TIME_BUDGET,
        }

        # Internal state — private, only exposed via state() deep copy
        self._inbox: List[Email] = []
        self._relationships: Dict[str, Relationship] = {}
        self._current_email_index: int = 0
        self._current_timestep: int = 0
        self._time_budget_remaining: int = DEFAULT_TIME_BUDGET
        self._total_time_spent: int = 0
        self._emails_handled: int = 0
        self._ignored_senders: Dict[str, int] = {}

        # Load email templates
        self._email_bank = self._load_email_bank()

    # -----------------------------------------------------------------------
    # PUBLIC INTERFACE
    # -----------------------------------------------------------------------

    def reset(self) -> EmailObservation:
        """
        Start a new episode.

        1. Sample emails from bank (VIP/Normal/Spam mix per task config)
        2. Shuffle inbox order
        3. Initialise all relationship healths at 75
        4. Reset all counters
        5. Return first email as Observation
        """
        cfg = self.task_config

        vip_pool    = self._email_bank.get("vip_emails", [])
        normal_pool = self._email_bank.get("normal_emails", [])
        spam_pool   = self._email_bank.get("spam_emails", [])

        vip_emails    = random.sample(vip_pool,    min(cfg["vip_count"],    len(vip_pool)))
        normal_emails = random.sample(normal_pool, min(cfg["normal_count"], len(normal_pool)))
        spam_emails   = random.sample(spam_pool,   min(cfg["spam_count"],   len(spam_pool)))

        all_emails = vip_emails + normal_emails + spam_emails
        random.shuffle(all_emails)

        # Build Email objects — assign sequential IDs for this episode
        self._inbox = []
        for i, template in enumerate(all_emails):
            email = Email(
                email_id               = i,
                sender                 = template["sender"],
                sender_domain          = template.get("sender_domain", "external"),
                subject                = template["subject"],
                body                   = template["body"],
                sender_importance      = template["sender_importance"],
                base_priority          = template["base_priority"],
                estimated_response_time= self._sample_time_cost(template),
                is_followup            = template.get("is_followup", False),
                parent_email_id        = template.get("parent_email_id", None),
                received_at_timestep   = 0,
            )
            self._inbox.append(email)

        # Initialise relationships for all unique senders
        self._relationships = {}
        for email in self._inbox:
            if email.sender not in self._relationships:
                cfg_rel = RELATIONSHIP_CONFIG[email.sender_importance]
                self._relationships[email.sender] = Relationship(
                    sender_email     = email.sender,
                    health           = INITIAL_RELATIONSHIP_HEALTH,
                    importance       = email.sender_importance,
                    importance_weight= cfg_rel["importance_weight"],
                    degradation_rate = abs(cfg_rel["ignore_penalty"]),
                )

        # Reset counters
        self._current_email_index   = 0
        self._current_timestep      = 0
        self._time_budget_remaining = self.task_config.get("time_budget", DEFAULT_TIME_BUDGET)
        self._total_time_spent      = 0
        self._emails_handled        = 0
        self._ignored_senders       = {}

        return self._build_observation()

    def step(self, action: int) -> StepResponse:
        """
        Execute one action and advance the environment.

        Args:
            action: 0 = IGNORE, 1 = RESPOND

        Returns:
            StepResponse(observation, reward, done, info)
        """
        # Guard against stepping after episode ended
        if self._current_email_index >= len(self._inbox):
            return StepResponse(
                observation=self._build_final_observation(),
                reward=0.0,
                done=True,
                info={"error": "step() called after episode ended"},
            )

        current_email = self._inbox[self._current_email_index]
        relationship  = self._relationships[current_email.sender]

        # ------ Calculate reward ------
        if action == Action.RESPOND:
            reward, time_cost = self._reward_respond(current_email, relationship)

            # Update time budget
            self._time_budget_remaining -= time_cost
            self._total_time_spent      += time_cost

            # Boost relationship health
            boost = RELATIONSHIP_CONFIG[current_email.sender_importance]["respond_boost"]
            relationship.health = min(100.0, relationship.health + boost)
            relationship.interaction_count += 1

            info = {
                "action":             "respond",
                "time_cost":          time_cost,
                "relationship_delta": boost,
                "reward":             reward,
            }

        else:  # IGNORE
            reward  = self._reward_ignore(current_email, relationship)
            penalty = RELATIONSHIP_CONFIG[current_email.sender_importance]["ignore_penalty"]

            # Degrade relationship health
            relationship.health = max(0.0, relationship.health + penalty)  # penalty is negative

            # Track ignore escalation
            sender = current_email.sender
            self._ignored_senders[sender] = self._ignored_senders.get(sender, 0) + 1
            current_email.times_ignored             += 1
            current_email.sender_urgency_multiplier += 0.5

            # Extra repeat-ignore penalty
            if self._ignored_senders[sender] > 1:
                relationship.health = max(0.0, relationship.health - 10)

            # Mark VIP as angry (would trigger follow-up injection)
            if current_email.sender_importance == "VIP":
                relationship.is_angry = True
                # Inject follow-up email into inbox
                followup_pool = [
                    e for e in self._email_bank.get("vip_emails", [])
                    if e.get("is_followup") and e.get("parent_email_id") == current_email.email_id
                ]
                if followup_pool:
                    template = followup_pool[0]
                    followup = Email(
                        email_id=len(self._inbox),
                        sender=template["sender"],
                        sender_domain=template.get("sender_domain", "internal"),
                        subject=template["subject"],
                        body=template["body"],
                        sender_importance="VIP",
                        base_priority=min(10, current_email.base_priority + 2),
                        estimated_response_time=self._sample_time_cost(template),
                        is_followup=True,
                        parent_email_id=current_email.email_id,
                        received_at_timestep=self._current_timestep,
                    )
                    # Insert after current position so agent sees it soon
                    insert_pos = min(
                        self._current_email_index + 2,
                        len(self._inbox)
                    )
                    self._inbox.insert(insert_pos, followup)

            info = {
                "action":             "ignore",
                "ignore_count":       self._ignored_senders.get(sender, 0),
                "relationship_delta": penalty,
                "reward":             reward,
            }

        self._emails_handled        += 1
        self._current_email_index   += 1
        self._current_timestep      += 1

        inbox_empty = self._current_email_index >= len(self._inbox)
        time_up     = self._time_budget_remaining <= 0
        done        = inbox_empty or time_up

        # Sunset penalty — time ran out, emails remain
        if done and time_up and not inbox_empty:
            sunset_penalty     = self._calculate_sunset_penalty()
            reward            += sunset_penalty
            info["sunset_penalty"] = sunset_penalty

        # Time bonus — cleared entire inbox before time ran out
        if done and inbox_empty:
            original_budget   = self.task_config.get("time_budget", DEFAULT_TIME_BUDGET)
            time_ratio        = self._time_budget_remaining / original_budget
            time_bonus        = round(time_ratio * 10, 4)
            reward           += time_bonus
            info["time_bonus"]     = time_bonus

        next_obs = self._build_final_observation() if done else self._build_observation()

        return StepResponse(
            observation=next_obs,
            reward=round(reward, 4),
            done=done,
            info=info,
        )

    def state(self) -> State:
        """
        Returns the full internal state as a DEEP COPY.
        Deep copy ensures the grader records frozen snapshots, not live references.
        """
        return State(
            inbox                 = copy.deepcopy(self._inbox),
            current_email_index   = self._current_email_index,
            relationships         = copy.deepcopy(self._relationships),
            current_timestep      = self._current_timestep,
            time_budget_remaining = self._time_budget_remaining,
            total_time_spent      = self._total_time_spent,
            emails_handled        = self._emails_handled,
        )

    # -----------------------------------------------------------------------
    # PRIVATE HELPERS
    # -----------------------------------------------------------------------

    def _reward_respond(self, email: Email, relationship: Relationship) -> Tuple[float, int]:
        """
        Reward formula for RESPOND action.
        reward = (email_value - 5.0 × normalized_cost) × (relationship_health / 100)

        email_value     = base_priority × urgency_multiplier
        normalized_cost = estimated_response_time / 120.0
            (so a 120-min email costs 1.0 in normalized units)
            (a 60-min email costs 0.5, a 5-min email costs ~0.04)

        Returns: (reward, time_cost_minutes)
        """
        email_value     = email.base_priority * email.sender_urgency_multiplier
        action_cost     = email.estimated_response_time
        normalized_cost = action_cost / 120.0

        reward = (email_value - 5.0 * normalized_cost) * (relationship.health / 100)
        return round(reward, 4), action_cost

    def _reward_ignore(self, email: Email, relationship: Relationship) -> float:
        """
        Reward formula for IGNORE action.
        Spam: 0 reward (correct decision).
        Others: -1 × (email_value × health_penalty / 100)
        """
        if email.sender_importance == "Spam":
            return 0.0

        health_penalty = 15 * relationship.importance_weight
        email_value    = email.base_priority * email.sender_urgency_multiplier
        reward         = -1 * (email_value * health_penalty / 100)
        return round(reward, 4)

    def _calculate_sunset_penalty(self) -> float:
        """
        Penalty when time runs out with emails remaining.
        FinalPenalty = -Σ(base_priority × relationship_health / 100) for remaining emails
        """
        penalty = 0.0
        for i in range(self._current_email_index, len(self._inbox)):
            email = self._inbox[i]
            rel   = self._relationships.get(email.sender)
            if rel:
                penalty -= email.base_priority * (rel.health / 100)
        return round(penalty, 4)

    def _build_observation(self) -> EmailObservation:
        """Build agent's limited view of the current email."""
        if self._current_email_index >= len(self._inbox):
            return self._build_final_observation()

        email    = self._inbox[self._current_email_index]
        rel      = self._relationships.get(email.sender)
        rel_score = rel.health if rel else 50.0
        emails_remaining = len(self._inbox) - self._current_email_index - 1

        return EmailObservation(
            email_id              = email.email_id,
            sender                = email.sender,
            subject               = email.subject,
            body                  = email.body,
            sender_importance     = email.sender_importance,
            email_length          = len(email.body),    # Proxy — NOT actual time cost
            relationship_score    = rel_score,
            time_budget_remaining = self._time_budget_remaining,
            emails_remaining      = emails_remaining,
        )

    def _build_final_observation(self) -> EmailObservation:
        """Dummy observation returned when episode is done."""
        return EmailObservation(
            email_id              = -1,
            sender                = "done@done.com",
            subject               = "Episode Complete",
            body                  = "",
            sender_importance     = "Spam",
            email_length          = 0,
            relationship_score    = 0.0,
            time_budget_remaining = self._time_budget_remaining,
            emails_remaining      = 0,
        )

    def _sample_time_cost(self, template: dict) -> int:
        """
        Returns estimated_response_time for an email.
        If the template has an explicit value, use it (capped at 180).
        Otherwise, sample from distribution:
            60% Quick      (5-15 min)
            30% Deep Work  (30-60 min)
            10% Black Hole (120 min)

        A VIP can get a Black Hole email. If the agent spends 2 hours on
        the founder's coffee preference email, it earns a negative reward.
        This forces the agent to read CONTENT, not just sender tier.
        """
        if "estimated_response_time" in template:
            return min(180, template["estimated_response_time"])

        roll = random.random()
        if roll < 0.6:
            return random.randint(5, 15)
        elif roll < 0.9:
            return random.randint(30, 60)
        else:
            return 120

    def _load_email_bank(self) -> dict:
        """Load pre-written email templates from data/email_bank.json."""
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "..", "data", "email_bank.json"),
            os.path.join(os.path.dirname(__file__), "data", "email_bank.json"),
            "data/email_bank.json",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, "r") as f:
                    return json.load(f)

        # Minimal fallback so env still runs without data file
        return {
            "vip_emails": [
                {
                    "sender": "boss@company.com", "sender_domain": "internal",
                    "subject": "Q4 Report Review Needed",
                    "body": "Please review the attached Q4 report and share feedback by EOD.",
                    "sender_importance": "VIP", "base_priority": 9,
                    "estimated_response_time": 45, "is_followup": False, "parent_email_id": None
                },
                {
                    "sender": "cto@company.com", "sender_domain": "internal",
                    "subject": "Architecture Decision Needed",
                    "body": "We need a decision on the new microservices architecture. Please review and respond.",
                    "sender_importance": "VIP", "base_priority": 8,
                    "estimated_response_time": 60, "is_followup": False, "parent_email_id": None
                },
            ],
            "normal_emails": [
                {
                    "sender": "teammate@company.com", "sender_domain": "internal",
                    "subject": "PR Review Request",
                    "body": "Hey, can you review my pull request when you get a chance? It's a small fix.",
                    "sender_importance": "Normal", "base_priority": 5,
                    "estimated_response_time": 15, "is_followup": False, "parent_email_id": None
                },
            ],
            "spam_emails": [
                {
                    "sender": "newsletter@random.com", "sender_domain": "external",
                    "subject": "Top 10 Productivity Hacks!",
                    "body": "Unsubscribe at any time. Limited offer today only!",
                    "sender_importance": "Spam", "base_priority": 1,
                    "estimated_response_time": 5, "is_followup": False, "parent_email_id": None
                },
            ],
        }