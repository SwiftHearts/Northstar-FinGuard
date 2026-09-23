"""Executive summary agent: produces a concise, CCO-ready summary of the compliance review --
overall severity, per-claim breakdown, and recommended next steps -- from the full pipeline state.

Usage (standalone test):
    python -m src.agents.executive_summary_agent
"""

import json

from src.graph.state import ComplianceState
from src.utils.azure_clients import get_azure_openai_client
from src.utils.config import settings

SUMMARY_PROMPT = """You are drafting an executive summary for the Chief Compliance Officer of Northstar \
Wealth Partners, summarizing an automated compliance review of a draft marketing/client communication.

Draft metadata: {metadata}

Document-level severity: {review_tier}
Routing: {routing_notes}

Per-claim findings:
{claim_findings}

Write a concise executive summary (4-6 sentences) covering: the overall risk level, the most \
significant finding(s) and why they matter, and the recommended next action. Write in plain prose, \
no headers or bullet points, suitable to paste directly into a compliance review ticket.
"""


def _format_claim_findings(claims: list[dict], risk_assessments: dict) -> str:
    lines = []
    for claim in claims:
        assessment = risk_assessments.get(claim["claim_id"], {})
        lines.append(
            f'- "{claim["text"]}" ({claim["claim_type"]}) -> '
            f'{assessment.get("tier", "unassessed")}: {assessment.get("rationale", "")}'
        )
    return "\n".join(lines) if lines else "(No claims were flagged.)"


def executive_summary_agent(state: ComplianceState) -> dict:
    client = get_azure_openai_client()

    prompt = SUMMARY_PROMPT.format(
        metadata=state.get("metadata", {}),
        review_tier=state.get("review_tier", "Tier 1"),
        routing_notes=state.get("routing_notes", ""),
        claim_findings=_format_claim_findings(state.get("claims", []), state.get("risk_assessments", {})),
    )

    response = client.chat.completions.create(
        model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return {"executive_summary": response.choices[0].message.content.strip()}


if __name__ == "__main__":
    sample_state = {
        "metadata": {"draft_id": "abc123", "channel": "social_media", "audience": "retail", "word_count": 26},
        "claims": [{"claim_id": "test1", "text": "beat the market every year", "claim_type": "performance"}],
        "risk_assessments": {
            "test1": {
                "tier": "Tier 3",
                "rationale": "Uses prohibited absolute performance language.",
                "cited_doc_id": "NWP-MKT-03",
            }
        },
        "review_tier": "Tier 3",
        "routing_notes": "Document-level severity: Tier 3, driven by 1 claim(s). "
        "Route to: Chief Compliance Officer (CCO).",
    }
    result = executive_summary_agent(sample_state)
    print(json.dumps(result, indent=2))
