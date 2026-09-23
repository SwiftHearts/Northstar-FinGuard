"""Review routing agent: aggregates claim-level risk tiers into a single document-level
severity and routes it per the Regulatory Review Playbook (NWP-RRP-07 Section 4) escalation
workflow and Section 5 service-level targets. Purely rule-based -- routing is deterministic
once each claim's tier is known, so no LLM call is needed here.

Usage (standalone test):
    python -m src.agents.review_routing_agent
"""

import json

from src.graph.state import ComplianceState

TIER_ORDER = ["Tier 1", "Tier 2", "Tier 3", "Tier 4"]

ROUTING = {
    "Tier 1": {
        "destination": "Content author",
        "acknowledgment": "Same business day",
        "remediation_target": "Before publication",
    },
    "Tier 2": {
        "destination": "Content author, then Compliance Reviewer sign-off",
        "acknowledgment": "1 business day",
        "remediation_target": "3 business days",
    },
    "Tier 3": {
        "destination": "Chief Compliance Officer (CCO)",
        "acknowledgment": "Same business day",
        "remediation_target": "5 business days",
    },
    "Tier 4": {
        "destination": "CCO and senior management (concurrent notification)",
        "acknowledgment": "Immediate (within hours)",
        "remediation_target": "24 hours for takedown; full remediation plan within 5 business days",
    },
}


def review_routing_agent(state: ComplianceState) -> dict:
    risk_assessments = state.get("risk_assessments", {})

    if not risk_assessments:
        return {"review_tier": "Tier 1", "routing_notes": "No claims required compliance review."}

    highest_tier = max((a["tier"] for a in risk_assessments.values()), key=TIER_ORDER.index)
    routing = ROUTING[highest_tier]

    driving_claim_ids = [claim_id for claim_id, a in risk_assessments.items() if a["tier"] == highest_tier]

    notes = (
        f"Document-level severity: {highest_tier}, driven by {len(driving_claim_ids)} claim(s). "
        f"Route to: {routing['destination']}. "
        f"Acknowledgment: {routing['acknowledgment']}. "
        f"Remediation target: {routing['remediation_target']}."
    )

    return {"review_tier": highest_tier, "routing_notes": notes}


if __name__ == "__main__":
    sample_risk = {
        "test1": {
            "tier": "Tier 3",
            "rationale": "Uses prohibited absolute performance language.",
            "cited_doc_id": "NWP-MKT-03",
        },
        "test2": {
            "tier": "Tier 2",
            "rationale": "Superlative lacks a citation to a ranking source.",
            "cited_doc_id": "NWP-MKT-03",
        },
    }
    result = review_routing_agent({"risk_assessments": sample_risk})
    print(json.dumps(result, indent=2))
