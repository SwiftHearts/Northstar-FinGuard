# Module docstring for src/graph/state.py
"""Shared state schema for the Northstar-FinGuard compliance-review LangGraph pipeline."""

# Description of the state (shared memory) used in the Northstar-FinGuard compliance-review LangGraph pipeline. 
# This state is structured as a TypedDict, which allows for type checking and validation of the data 
# passed between different agents in the pipeline. Each agent can read from and write to this shared state, 
# enabling a coordinated workflow for processing compliance review tasks.

# From typing import Literal, TypedDict, which are used to define structured types for the 
# compliance review state. These types help ensure that the data passed between different agents 
# in the pipeline adheres to a consistent schema, making it easier to manage and validate the state 
# throughout the compliance review process.
from typing import Literal, TypedDict

# Defining exactly (literal) the allowed values for the channel, audience, and claim type fields in the state schema.
Channel = Literal["email", "social_media", "website", "brochure", "pitch_deck", "webinar", "other"]
Audience = Literal["retail", "institutional", "mixed"]
ClaimType = Literal[
    "performance",
    "guarantee_or_risk_free",
    "comparative",
    "superlative",
    "testimonial_endorsement",
    "fee_or_pricing",
    "regulatory_endorsement_implication",
    "general_factual",
]

# Defining the structure of a claim in the compliance review state as a Typed Dictionary. 
class Claim(TypedDict):
    claim_id: str
    text: str
    claim_type: ClaimType


class IntakeMetadata(TypedDict):
    draft_id: str
    channel: Channel
    audience: Audience
    word_count: int

# Shared state passed between agents 
# Total=False allows for optional fields, meaning that not all fields need to be present in the state at all times.
class ComplianceState(TypedDict, total=False):
    # --- Intake (Agent 1) ---
    # Original document
    source_text: str
    metadata: IntakeMetadata

    # --- Claim extraction (Agent 2) ---
    # Agent extracts claims from the source text and stores them in a list of Claim dictionaries.
    claims: list[Claim]

    # --- Evidence retrieval (Agent 3) ---
    # Agent maps each claim to a list of evidence items, where each evidence item is represented as a dictionary.
    evidence: dict[str, list[dict]]

    # --- Grounding verification (Agent 4) ---
    # Agent checks whether each claim is grounded in the provided evidence and stores the results in a dictionary.
    grounding_results: dict[str, dict]

    # --- Compliance risk (Agent 5) ---
    # Agent assesses the compliance risk for each claim and stores the results in a dictionary.
    risk_assessments: dict[str, dict]

    # --- Review routing (Agent 6) ---
    # Agent determines the appropriate review tier and any routing notes for each claim, storing the 
    # results in the state.
    review_tier: str
    routing_notes: str

    # --- Executive summary (Agent 7) ---
    # Agent generates an executive summary of the compliance review findings and stores it in the state.
    executive_summary: str
