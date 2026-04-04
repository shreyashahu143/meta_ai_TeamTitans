"""
models.py — Type-safe data contracts for the Email Triage RL Environment
=========================================================================

PURPOSE:
    This is the "Menu" of our restaurant analogy.
    It defines EXACTLY what data looks like when flowing between:
      - server/environment.py  (the kitchen that creates data)
      - client.py              (the waiter that carries data)
      - inference.py           (the AI that reads data)

    Everyone imports from here. Change here → change everywhere.

CONNECTS TO:
    → server/environment.py  (uses Email, State, Relationship to manage game state)
    → server/app.py          (serializes/deserializes these models to JSON)
    → client.py              (uses EmailObservation, StepResponse, State)
    → inference.py           (reads EmailObservation to make decisions)
    → grader.py              (reads full State + episode history for scoring)

HOW TO USE:
    from models import Email, State, EmailObservation, StepResponse, SenderImportance
"""

from __future__ import annotations

from enum import IntEnum , Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. ENUMS
# ---------------------------------------------------------------------------

class SenderImportance(str ,Enum):
    """
    The "tier" of a sender. Controls relationship decay rates and reward weights.

    VIP    → Boss, Team Lead, Key Client   (ignore_penalty = -20, respond_boost = +15)
    NORMAL → Teammates, Collaborators      (ignore_penalty = -10, respond_boost = +10)
    SPAM   → Newsletters, Cold Outreach    (ignore_penalty =  0,  respond_boost =  0)
    """
    VIP = "VIP"
    NORMAL = "Normal"
    SPAM = "Spam"


class Action(IntEnum):
    """
    The two actions the AI agent can take each step.

    IGNORE  (0) → Skip this email. Saves time but damages relationship.
    RESPOND (1) → Reply to this email. Costs time but boosts relationship.
    """
    IGNORE = 0
    RESPOND = 1

    # CHANGE TO:
    # Action type is just int (0 or 1)
    # No need for Enum — OpenEnv expects raw int


# ---------------------------------------------------------------------------
# 2. EMAIL
# ---------------------------------------------------------------------------

class Email(BaseModel):
    """
    Represents a single email in the inbox.

    KEY DESIGN DECISIONS:
    - `base_priority` (1-10): The email's inherent importance. Does NOT always
       correlate with sender_importance (a VIP can send a low-priority "lunch?" email).
    - `estimated_response_time`: Hidden from the agent (it only sees email_length).
       This is Blind Spot #1 — agent must INFER cost from content, not read it directly.
    - `times_ignored`: Tracks how many times this sender has been ignored.
       On ignore, sender_urgency_multiplier increases by +0.5 each time.
    - `is_followup`: True if this email was generated BECAUSE a previous email was ignored.
       Follow-ups get a +20 priority boost automatically.
    """

    email_id: int
    sender: str = Field(..., description="Full email address, e.g. boss@company.com")
    sender_domain: str = Field(..., description="'internal', 'external', or 'vip-domain'")
    subject: str
    body: str = Field(..., description="Full email body text")

    # Importance & Priority
    sender_importance: str = Field(..., description="VIP | Normal | Spam")
    base_priority: int = Field(..., ge=1, le=10, description="Inherent priority 1-10")

    # Time tracking (Blind Spot #1)
    estimated_response_time: int = Field(
        ..., ge=5, le=120,
        description="Minutes to respond. HIDDEN from agent — it sees email_length instead."
    )

    # Episode tracking
    received_at_timestep: int = Field(default=0, description="Which step this arrived")
    times_ignored: int = Field(default=0, description="How many times ignored so far")
    sender_urgency_multiplier: float = Field(
        default=1.0,
        description="Escalates by +0.5 each time this sender is ignored"
    )

    # Follow-up flag (Blind Spot #4 implementation)
    is_followup: bool = Field(
        default=False,
        description="True if generated because a previous email from this sender was ignored"
    )
    #added by shreya [i'll write sss to  indicate this chnage is made by me ]
    parent_email_id: Optional[int] = Field(
        default=None,
        description="If this is a followup, the email_id of the original email"
    )


# ---------------------------------------------------------------------------
# 3. RELATIONSHIP
# ---------------------------------------------------------------------------

class Relationship(BaseModel):
    """
    Tracks the health of our working relationship with each sender.

    health:           Current score (0-100). Starts at 75 for all senders.
    importance_weight: Maps sender_importance to a multiplier for penalty calculations.
                       VIP=3, Normal=2, Spam=0
    degradation_rate: How many health points lost per ignore (set by RELATIONSHIP_CONFIG).
    is_angry:         Set to True when a VIP is ignored. Triggers follow-up email generation.
    interaction_count: Total number of times we've interacted (responded to) this sender.

    DECAY TABLE (defined in server/environment.py as RELATIONSHIP_CONFIG):
        VIP:    ignore_penalty=-20, respond_boost=+15, followup_multiplier=1.5
        Normal: ignore_penalty=-10, respond_boost=+10, followup_multiplier=1.2
        Spam:   ignore_penalty=0,   respond_boost=0,   followup_multiplier=1.0
    """

    sender_email: str
    health: float = Field(default=75.0, ge=0.0, le=100.0)
    importance: str = Field(..., description="VIP | Normal | Spam")
    importance_weight: int = Field(
        ..., ge=0, le=3,
        description="VIP=3, Normal=2, Spam=0. Used in health_penalty formula."
    )
    degradation_rate: float = Field(
        ..., description="Health lost per ignore. Pulled from RELATIONSHIP_CONFIG."
    )
    is_angry: bool = Field(
        default=False,
        description="True if this sender was ignored. Next email from them gets +20 priority."
    )
    interaction_count: int = Field(default=0, description="How many times we responded")


# ---------------------------------------------------------------------------
# 4. STATE  (God-mode view — used by grader and for debugging)
# ---------------------------------------------------------------------------

class State(BaseModel):
    """
    Complete snapshot of the environment.
    The grader uses this to check if the AI actually managed relationships correctly.

    IMPORTANT: environment.py must return a DEEP COPY (not references to live objects).
    Use copy.deepcopy() before returning. If you return live references, grader logs
    will show the FINAL state for every step, not the state at each step.

    CONNECTS TO:
        → server/environment.py: state() method creates and returns this
        → server/app.py: GET /state endpoint serializes this to JSON
        → client.py: _parse_state() deserializes JSON into this model
        → grader.py: episode_history stores a list of these State objects
    """

    inbox: List[Email] = Field(..., description="All emails in this episode (processed or not)")
    current_email_index: int = Field(
        default=0, description="Index into inbox. Points to the CURRENT email to act on."
    )
    relationships: Dict[str, Relationship] = Field(
        ..., description="Maps sender_email → Relationship object"
    )
    current_timestep: int = Field(default=0)
    time_budget_remaining: int = Field(
        default=480, description="Minutes left in the workday. Starts at 480 (8 hours)."
    )
    total_time_spent: int = Field(default=0, description="Cumulative minutes spent responding")
    emails_handled: int = Field(default=0, description="Count of emails processed so far")


# ---------------------------------------------------------------------------
# 5. OBSERVATION  (What the agent actually sees — partial view)
# ---------------------------------------------------------------------------

class EmailObservation(BaseModel):
    """
    What the LLM agent sees at each step. A LIMITED view of State.

    KEY HIDDEN INFORMATION (intentional partial observability):
    - Agent does NOT see: estimated_response_time (only email_length as proxy)
    - Agent does NOT see: other senders' relationship scores (only current sender's)
    - Agent does NOT see: future emails in the inbox
    - Agent does NOT see: relationship degradation rates

    This is what makes it a real RL problem: the agent must INFER cost from length
    and INFER urgency from relationship_score — not read them directly.

    CONNECTS TO:
        → server/environment.py: _build_observation() creates this from current Email + State
        → server/app.py: POST /step returns this as part of StepResponse JSON
        → client.py: _parse_result() deserializes JSON → EmailObservation
        → inference.py: LLM reads this to decide action (0 or 1)
    """

    email_id: int
    sender: str
    subject: str
    body: str
    sender_importance: str = Field(..., description="VIP | Normal | Spam (visible to agent)")
    email_length: int = Field(..., description="Character count. Proxy for time cost.")
    relationship_score: float = Field(
        ..., description="Current health with THIS sender only (0-100)"
    )
    time_budget_remaining: int = Field(..., description="Minutes left in workday")
    emails_remaining: int = Field(..., description="How many emails are left to process")


# ---------------------------------------------------------------------------
# 6. STEP RESPONSE  (What step() returns via API)
# ---------------------------------------------------------------------------

class StepResponse(BaseModel):
    """
    The full response from POST /step.
    Wraps everything the agent needs after taking an action.

    CONNECTS TO:
        → server/app.py: POST /step returns this as JSON
        → client.py: _parse_result() extracts .observation from this
        → inference.py: reads .reward and .done to track episode progress
    """

    observation: EmailObservation = Field(
        ..., description="The NEXT email to act on (or final observation if done=True)"
    )
    reward: float = Field(..., description="Reward earned for the action just taken")
    done: bool = Field(..., description="True if episode is over (inbox empty or time ran out)")
    info: dict = Field(
        default_factory=dict,
        description="Extra debug info: action_taken, time_cost, relationship_delta, etc."
    )


# ---------------------------------------------------------------------------
# 7. RELATIONSHIP CONFIG (constants — imported by environment.py)
# ---------------------------------------------------------------------------

RELATIONSHIP_CONFIG = {
    "VIP": {
        "ignore_penalty": -20,
        "respond_boost": +15,
        "followup_multiplier": 1.5,
        "importance_weight": 3,
    },
    "Normal": {
        "ignore_penalty": -10,
        "respond_boost": +10,
        "followup_multiplier": 1.2,
        "importance_weight": 2,
    },
    "Spam": {
        "ignore_penalty": 0,
        "respond_boost": 0,
        "followup_multiplier": 1.0,
        "importance_weight": 0,
    },
}
