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
        close()  → optional cleanup (added for spec completeness)

REWARD FUNCTION:
    RESPOND (1):
        email_value     = base_priority × urgency_multiplier   (range: 1–10+)
        normalized_cost = estimated_response_time / 120.0      (range: 0–1.5 for 180-min emails)
        reward = (email_value - 5.0 × normalized_cost) × (relationship_health / 100)

    IGNORE (0):
        if Spam → reward = 0 (correct decision, no time penalty, no relationship penalty)
        else    → health_penalty = 15 × importance_weight
                  reward = -1 × (email_value × health_penalty / 100)
                  reading_cost = max(MIN_IGNORE_TIME, int(estimated_response_time × IGNORE_READING_TIME_FACTOR))
                  time_budget_remaining -= reading_cost

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

DEFAULT_TIME_BUDGET         = 480   # 8-hour workday in minutes
INITIAL_RELATIONSHIP_HEALTH = 75.0  # All senders start at 75/100

# Agents pay a reading cost even when they ignore an email.
# Without this, ignoring is zero-cost and the env is trivially exploitable.
IGNORE_READING_TIME_FACTOR  = 0.2   # Reading before ignoring costs 20% of response time
MIN_IGNORE_TIME             = 2     # Minimum 2 minutes to read any email before ignoring


class EmailTriageEnv:
    """
    The Email Triage RL Environment.

    One episode = one simulated workday.
    Each step = agent decides IGNORE (0) or RESPOND (1) to one email.

    Episode ends when:
        1. Inbox is empty (current_email_index >= len(inbox))  — clean finish
        2. Time budget runs out (time_budget_remaining <= 0)   — forced end
    """

    def __init__(self, task_config: Optional[dict] = None, seed: Optional[int] = None):
        """
        Args:
            task_config: Dict from tasks/task_X.json.
                         Controls num_emails, distribution, time_budget.
                         If None, uses Task 1 defaults.
            seed:        Random seed for reproducibility.
                         When provided, the FIRST episode is seeded deterministically.
                         Subsequent reset() calls advance the RNG normally so that
                         repeated runs produce different episodes (variance check passes).
                         To replay the exact same episode, pass the same seed to a
                         fresh EmailTriageEnv() instance — do NOT re-seed on reset().
        """
        self.task_config = task_config or {
            "num_emails":   20,
            "vip_count":    5,
            "normal_count": 10,
            "spam_count":   5,
            "time_budget":  DEFAULT_TIME_BUDGET,
        }

        # Seed once at construction time only.
        # BUG IN SUBMITTED VERSION: reset() re-seeded with the same seed every call,
        # making every episode identical. This would fail Phase 2 variance checks because
        # graders that always return the same score on identical episodes are flagged.
        # The seed is for reproducible FIRST episodes (e.g. unit tests), not for locking
        # every episode to the same shuffle.
        if seed is not None:
            random.seed(seed)

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

        NOTE: Does NOT re-seed the RNG. Each call to reset() advances the RNG
        from where it left off, producing a different episode each time.
        This is intentional — Phase 2 variance checks require different episodes
        to produce different scores. If you need a fully deterministic replay,
        create a new EmailTriageEnv(seed=N) instance instead.
        """
        cfg = self.task_config

        vip_pool    = self._email_bank.get("vip_emails", [])
        normal_pool = self._email_bank.get("normal_emails", [])
        spam_pool   = self._email_bank.get("spam_emails", [])

        vip_emails    = random.sample(vip_pool,    min(cfg["vip_count"],    len(vip_pool)))
        normal_emails = random.sample(normal_pool, min(cfg["normal_count"], len(normal_pool)))
        spam_emails   = random.sample(spam_pool,   min(cfg["spam_count"],   len(spam_pool)))

        # Warn when the bank is too small rather than silently under-sampling
        if len(vip_emails) < cfg["vip_count"]:
            print(f"[WARNING] VIP pool too small: requested {cfg['vip_count']}, got {len(vip_emails)}", flush=True)
        if len(normal_emails) < cfg["normal_count"]:
            print(f"[WARNING] Normal pool too small: requested {cfg['normal_count']}, got {len(normal_emails)}", flush=True)
        if len(spam_emails) < cfg["spam_count"]:
            print(f"[WARNING] Spam pool too small: requested {cfg['spam_count']}, got {len(spam_emails)}", flush=True)

        total_requested = cfg["vip_count"] + cfg["normal_count"] + cfg["spam_count"]
        total_sampled   = len(vip_emails) + len(normal_emails) + len(spam_emails)
        if total_sampled < total_requested:
            print(
                f"[WARNING] Episode will have {total_sampled} emails instead of "
                f"{total_requested} — fallback bank is active",
                flush=True,
            )

        all_emails = vip_emails + normal_emails + spam_emails
        random.shuffle(all_emails)

        # Build Email objects — assign sequential IDs for this episode
        self._inbox = []
        for i, template in enumerate(all_emails):
            email = Email(
                email_id                = i,
                sender                  = template["sender"],
                sender_domain           = template.get("sender_domain", "external"),
                subject                 = template["subject"],
                body                    = template["body"],
                sender_importance       = template["sender_importance"],
                base_priority           = template["base_priority"],
                estimated_response_time = self._sample_time_cost(template),
                is_followup             = template.get("is_followup", False),
                parent_email_id         = template.get("parent_email_id", None),
                received_at_timestep    = 0,
            )
            email.bank_email_id = template.get("email_id")  # store original bank ID
            self._inbox.append(email)

        # Initialise relationships for all unique senders
        self._relationships = {}
        for email in self._inbox:
            if email.sender not in self._relationships:
                cfg_rel = RELATIONSHIP_CONFIG[email.sender_importance]
                self._relationships[email.sender] = Relationship(
                    sender_email      = email.sender,
                    health            = INITIAL_RELATIONSHIP_HEALTH,
                    importance        = email.sender_importance,
                    importance_weight = cfg_rel["importance_weight"],
                    degradation_rate  = abs(cfg_rel["ignore_penalty"]),
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

        # ------ Calculate reward and time cost ------
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
            reward = self._reward_ignore(current_email, relationship)

            # Agents still pay a reading cost on ignore.
            # Spam is exempted — reading and binning spam is genuinely near-zero cost.
            if current_email.sender_importance != "Spam":
                reading_cost = max(
                    MIN_IGNORE_TIME,
                    int(current_email.estimated_response_time * IGNORE_READING_TIME_FACTOR),
                )
                self._time_budget_remaining -= reading_cost
                self._total_time_spent      += reading_cost

                # Apply relationship penalty
                penalty = RELATIONSHIP_CONFIG[current_email.sender_importance]["ignore_penalty"]
                relationship.health = max(0.0, relationship.health + penalty)  # penalty is negative

                # Track ignore escalation
                sender = current_email.sender
                self._ignored_senders[sender] = self._ignored_senders.get(sender, 0) + 1
                current_email.times_ignored             += 1
                current_email.sender_urgency_multiplier += 0.5

                # Extra repeat-ignore penalty
                if self._ignored_senders[sender] > 1:
                    relationship.health = max(0.0, relationship.health - 10)

                # Mark sender as angry and inject follow-up for VIP/Normal
                if current_email.sender_importance in ("VIP", "Normal"):
                    relationship.is_angry = True

                    # Guard prevents injecting a follow-up from a follow-up,
                    # which would create infinite email chains.
                    if not current_email.is_followup:
                        already_has_followup = any(
                            e.parent_email_id == current_email.email_id
                            for e in self._inbox
                        )
                        if not already_has_followup:
                            followup = Email(
                                email_id                = len(self._inbox),
                                sender                  = current_email.sender,
                                sender_domain           = current_email.sender_domain,
                                subject                 = f"FOLLOW UP: {current_email.subject}",
                                body                    = (
                                    f"I still need a response on my previous email:\n\n"
                                    f"---\n{current_email.body[:500]}\n---\n\n"
                                    f"Please reply as soon as possible."
                                ),
                                sender_importance       = current_email.sender_importance,
                                base_priority           = min(10, current_email.base_priority + 2),
                                estimated_response_time = max(5, current_email.estimated_response_time // 2),
                                is_followup             = True,
                                parent_email_id         = current_email.email_id,
                                received_at_timestep    = self._current_timestep,
                            )
                            insert_pos = min(self._current_email_index + 2, len(self._inbox))
                            self._inbox.insert(insert_pos, followup)

                info = {
                    "action":             "ignore",
                    "time_cost":          reading_cost,
                    "ignore_count":       self._ignored_senders.get(current_email.sender, 0),
                    "relationship_delta": penalty,
                    "reward":             reward,
                }

            else:
                # Spam: correct ignore — zero time cost, zero relationship damage
                info = {
                    "action":             "ignore",
                    "time_cost":          0,
                    "ignore_count":       0,
                    "relationship_delta": 0,
                    "reward":             reward,
                }

        self._emails_handled      += 1
        self._current_email_index += 1
        self._current_timestep    += 1

        inbox_empty = self._current_email_index >= len(self._inbox)
        time_up     = self._time_budget_remaining <= 0
        done        = inbox_empty or time_up

        # Mutually exclusive if/elif prevents double-counting the edge case
        # where the inbox empties at the exact same step time runs out.
        original_budget = self.task_config.get("time_budget", DEFAULT_TIME_BUDGET)

        if done:
            if time_up and not inbox_empty:
                # Sunset penalty — time ran out, emails remain
                sunset_penalty         = self._calculate_sunset_penalty()
                reward                += sunset_penalty
                info["sunset_penalty"] = sunset_penalty
            elif inbox_empty:
                # Time bonus — cleared entire inbox before time ran out
                time_ratio         = self._time_budget_remaining / original_budget
                time_bonus         = round(time_ratio * 10, 4)
                reward            += time_bonus
                info["time_bonus"] = time_bonus

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

    def close(self) -> None:
        """
        Clean up any resources. Added for OpenEnv spec completeness.
        Currently a no-op — extend if file handles or network sockets are added.

        TODO: Ensure server/app.py exposes a POST /close endpoint that calls
        env.close(), otherwise client.close_env() in inference.py silently does nothing.
        """
        pass

    # -----------------------------------------------------------------------
    # PRIVATE HELPERS
    # -----------------------------------------------------------------------

    def _reward_respond(self, email: Email, relationship: Relationship) -> Tuple[float, int]:
        """
        Reward formula for RESPOND action.
        reward = (email_value - 5.0 × normalized_cost) × (relationship_health / 100)

        email_value     = base_priority × urgency_multiplier
        normalized_cost = estimated_response_time / 120.0
            (120-min email → cost 1.0, 60-min → 0.5, 5-min → ~0.04)

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
        Spam: 0.0 — correct decision, no penalty.
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
        """Build the agent's limited, partial view of the current email."""
        if self._current_email_index >= len(self._inbox):
            return self._build_final_observation()

        email            = self._inbox[self._current_email_index]
        rel              = self._relationships.get(email.sender)
        rel_score        = rel.health if rel else 50.0
        emails_remaining = len(self._inbox) - self._current_email_index - 1

        return EmailObservation(
            email_id              = email.email_id,
            sender                = email.sender,
            subject               = email.subject,
            body                  = email.body,
            sender_importance     = email.sender_importance,
            email_length          = len(email.body),  # Proxy — NOT actual time cost
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
        Derives estimated_response_time from body length so the agent
        can actually learn the proxy signal it's given in observations.

        Buckets (with noise so it's learnable but not deterministic):
            < 150 chars   →  5–15  min  (quick reply)
            150–300 chars → 15–35  min  (standard)
            300–500 chars → 35–70  min  (deep work)
            > 500 chars   → 70–120 min  (black hole)
        """
        body_length = len(template.get("body", ""))

        if body_length < 150:
            return random.randint(5, 15)
        elif body_length < 300:
            return random.randint(15, 35)
        elif body_length < 500:
            return random.randint(35, 70)
        else:
            return random.randint(70, 120)

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

        # Minimal fallback so env still runs without data file.
        # TODO: Replace with a proper email_bank.json before submission.
        # This fallback has 2 VIP, 1 Normal, 1 Spam — too small for Task 2/3 configs.
        # The pool-size warnings in reset() will fire loudly when this is active.
        return {
            "vip_emails": [
                {
                    "sender": "boss@company.com", "sender_domain": "internal",
                    "subject": "Q4 Report Review Needed",
                    "body": "Please review the attached Q4 report and share feedback by EOD.",
                    "sender_importance": "VIP", "base_priority": 9,
                    "estimated_response_time": 45, "is_followup": False, "parent_email_id": None,
                },
                {
                    "sender": "cto@company.com", "sender_domain": "internal",
                    "subject": "Architecture Decision Needed",
                    "body": "We need a decision on the new microservices architecture. Please review and respond.",
                    "sender_importance": "VIP", "base_priority": 8,
                    "estimated_response_time": 60, "is_followup": False, "parent_email_id": None,
                },
            ],
            "normal_emails": [
                {
                    "sender": "teammate@company.com", "sender_domain": "internal",
                    "subject": "PR Review Request",
                    "body": "Hey, can you review my pull request when you get a chance? It's a small fix.",
                    "sender_importance": "Normal", "base_priority": 5,
                    "estimated_response_time": 15, "is_followup": False, "parent_email_id": None,
                },
            ],
            "spam_emails": [
                {
                    "sender": "newsletter@random.com", "sender_domain": "external",
                    "subject": "Top 10 Productivity Hacks!",
                    "body": "Unsubscribe at any time. Limited offer today only!",
                    "sender_importance": "Spam", "base_priority": 1,
                    "estimated_response_time": 5, "is_followup": False, "parent_email_id": None,
                },
            ],
        }