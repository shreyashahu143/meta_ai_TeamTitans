"""
models.py — Type-safe data contracts for the Email Triage RL Environment
=========================================================================

PURPOSE:
    Defines EXACTLY what data looks like when flowing between:
      - server/environment.py  (the kitchen that creates data)
      - client.py              (the waiter that carries data)
      - inference.py           (the AI that reads data)
      - grader.py              (the food critic that scores)

    Everyone imports from here. Change here → change everywhere.

CHANGES FROM ORIGINAL:
    - estimated_response_time max raised from 120 → 180
      (email_bank.json has a 180-min email; old cap caused Pydantic validation error)
    - All other fields unchanged
"""

from __future__ import annotations

from enum import IntEnum, Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. ENUMS
# ---------------------------------------------------------------------------

class SenderImportance(str, Enum):
    """
    The "tier" of a sender. Controls relationship decay rates and reward weights.

    VIP    → Boss, Team Lead, Key Client   (ignore_penalty = -20, respond_boost = +15)
    NORMAL → Teammates, Collaborators      (ignore_penalty = -10, respond_boost = +10)
    SPAM   → Newsletters, Cold Outreach    (ignore_penalty =  0,  respond_boost =  0)
    """
    VIP    = "VIP"
    NORMAL = "Normal"
    SPAM   = "Spam"


class Action(IntEnum):
    """
    The two actions the AI agent can take each step.

    IGNORE  (0) → Skip this email. Saves time but damages relationship.
    RESPOND (1) → Reply to this email. Costs time but boosts relationship.
    """
    IGNORE  = 0
    RESPOND = 1


# ---------------------------------------------------------------------------
# 2. EMAIL
# ---------------------------------------------------------------------------

class Email(BaseModel):
    """
    Represents a single email in the inbox.

    KEY DESIGN DECISIONS:
    - base_priority (1-10): inherent importance. NOT always correlated with
      sender_importance (a VIP can send a low-priority "lunch?" email).
    - estimated_response_time: HIDDEN from the agent — it sees email_length instead.
      This is Blind Spot #1.
    - times_ignored: tracks ignore count per email; urgency escalates.
    - is_followup: True if this email exists because a previous one was ignored.
    - parent_email_id: links follow-up to its original email.
    """

    email_id: int
    sender: str = Field(..., description="Full email address, e.g. boss@company.com")
    sender_domain: str = Field(..., description="'internal', 'external', or 'vip-domain'")
    subject: str
    body: str = Field(..., description="Full email body text")

    # Importance & Priority
    sender_importance: str = Field(..., description="VIP | Normal | Spam")
    base_priority: int = Field(..., ge=1, le=10, description="Inherent priority 1-10")

    # Time tracking (Blind Spot #1) — max 180 to support long audit emails
    estimated_response_time: int = Field(
        ..., ge=5, le=180,
        description="Minutes to respond. HIDDEN from agent — it sees email_length instead."
    )

    # Episode tracking
    received_at_timestep: int = Field(default=0)
    times_ignored: int = Field(default=0)
    sender_urgency_multiplier: float = Field(
        default=1.0,
        description="Escalates by +0.5 each time this sender is ignored"
    )

    # Follow-up tracking
    is_followup: bool = Field(
        default=False,
        description="True if generated because a previous email from this sender was ignored"
    )
    parent_email_id: Optional[int] = Field(
        default=None,
        description="email_id of the original email if this is a follow-up"
    )


# ---------------------------------------------------------------------------
# 3. RELATIONSHIP
# ---------------------------------------------------------------------------

class Relationship(BaseModel):
    """
    Tracks the health of our working relationship with each sender.

    health:            Current score (0-100). Starts at 75.
    importance_weight: VIP=3, Normal=2, Spam=0. Used in health_penalty formula.
    degradation_rate:  Health lost per ignore. Pulled from RELATIONSHIP_CONFIG.
    is_angry:          True if VIP was ignored. Triggers follow-up generation.
    interaction_count: Count of times we responded to this sender.

    DECAY TABLE (from RELATIONSHIP_CONFIG below):
        VIP:    ignore_penalty=-20, respond_boost=+15
        Normal: ignore_penalty=-10, respond_boost=+10
        Spam:   ignore_penalty=0,   respond_boost=0
    """

    sender_email: str
    health: float = Field(default=75.0, ge=0.0, le=100.0)
    importance: str = Field(..., description="VIP | Normal | Spam")
    importance_weight: int = Field(..., ge=0, le=3)
    degradation_rate: float = Field(...)
    is_angry: bool = Field(default=False)
    interaction_count: int = Field(default=0)


# ---------------------------------------------------------------------------
# 4. STATE  (God-mode view — used by grader and for debugging)
# ---------------------------------------------------------------------------

class State(BaseModel):
    """
    Complete snapshot of the environment.

    IMPORTANT: environment.py must return a DEEP COPY (not live references).
    Use copy.deepcopy() before returning.
    """

    inbox: List[Email] = Field(..., description="All emails in this episode")
    current_email_index: int = Field(default=0)
    relationships: Dict[str, Relationship] = Field(
        ..., description="Maps sender_email → Relationship object"
    )
    current_timestep: int = Field(default=0)
    time_budget_remaining: int = Field(default=480)
    total_time_spent: int = Field(default=0)
    emails_handled: int = Field(default=0)


# ---------------------------------------------------------------------------
# 5. OBSERVATION  (What the agent actually sees — partial view)
# ---------------------------------------------------------------------------

class EmailObservation(BaseModel):
    """
    What the LLM agent sees at each step. A LIMITED view of State.

    HIDDEN from agent (intentional partial observability):
      - estimated_response_time (agent sees email_length as proxy only)
      - other senders' relationship scores
      - future emails in the inbox
      - relationship degradation rates
    """

    email_id: int
    sender: str
    subject: str
    body: str
    sender_importance: str = Field(..., description="VIP | Normal | Spam")
    email_length: int = Field(..., description="Character count. Proxy for time cost.")
    relationship_score: float = Field(..., description="Health with THIS sender only (0-100)")
    time_budget_remaining: int = Field(...)
    emails_remaining: int = Field(...)


# ---------------------------------------------------------------------------
# 6. STEP RESPONSE
# ---------------------------------------------------------------------------

class StepResponse(BaseModel):
    """
    The full response from POST /step.
    """

    observation: EmailObservation = Field(
        ..., description="The NEXT email to act on (or final observation if done=True)"
    )
    reward: float = Field(...)
    done: bool = Field(...)
    info: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# 7. RELATIONSHIP CONFIG (constants — imported by environment.py)
# ---------------------------------------------------------------------------

RELATIONSHIP_CONFIG = {
    "VIP": {
        "ignore_penalty":      -20,
        "respond_boost":       +15,
        "followup_multiplier": 1.5,
        "importance_weight":   3,
    },
    "Normal": {
        "ignore_penalty":      -10,
        "respond_boost":       +10,
        "followup_multiplier": 1.2,
        "importance_weight":   2,
    },
    "Spam": {
        "ignore_penalty":      0,
        "respond_boost":       0,
        "followup_multiplier": 1.0,
        "importance_weight":   0,
    },
}