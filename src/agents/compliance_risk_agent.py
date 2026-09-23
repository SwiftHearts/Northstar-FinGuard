"""Compliance risk agent: for each claim and its reranked evidence, uses the LLM to determine
whether the claim violates Northstar policy and assigns a severity tier per the Regulatory
Review Playbook (NWP-RRP-07 Section 3): Tier 1 (Low) through Tier 4 (Critical).

Usage (standalone test):
    python -m src.agents.compliance_risk_agent
"""

import json

from src.graph.state import ComplianceState
from src.utils.azure_clients import get_azure_openai_client
from src.utils.config import settings

RISK_TIERS = ["Tier 1", "Tier 2", "Tier 3", "Tier 4"]

RISK_PROMPT = """You are a compliance reviewer at Northstar Wealth Partners, a registered investment \
adviser, applying the firm's Regulatory Review Playbook (NWP-RRP-07) severity tiers:

- Tier 1 (Low): stylistic or minor disclosure formatting issues; no escalation required.
- Tier 2 (Moderate): a claim lacks full substantiation, or a required disclosure is missing, but the \
underlying claim is not overtly misleading.
- Tier 3 (High): the claim uses prohibited language (per NWP-MKT-03 Section 6) or misrepresents risk, \
fees, or performance in a way likely to mislead a reasonable investor.
- Tier 4 (Critical): a materially false or misleading statement, an unauthorized guarantee, or a \
fiduciary duty breach.

Claim under review: "{claim_text}"
Claim type: {claim_type}

Relevant policy excerpts retrieved from the knowledge base (may not all be relevant -- use your \
judgment on which, if any, actually apply):
{evidence_block}

Assess this claim. Return strict JSON with exactly these keys:
- "tier": one of {tiers}
- "rationale": a one to two sentence explanation citing the specific policy provision that applies
- "cited_doc_id": the doc_id of the most relevant excerpt above, or null if none genuinely apply
"""


def _format_evidence(chunks: list[dict]) -> str:
    if not chunks:
        return "(No candidate policy language was retrieved for this claim.)"

    return "\n\n".join(f"[{c['doc_id']} - {c['section_title']}]\n{c['content']}" for c in chunks)


def _assess(claim: dict, evidence_chunks: list[dict]) -> dict:
    client = get_azure_openai_client()
    prompt = RISK_PROMPT.format(
        claim_text=claim["text"],
        claim_type=claim["claim_type"],
        evidence_block=_format_evidence(evidence_chunks),
        tiers=RISK_TIERS,
    )

    response = client.chat.completions.create(
        model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    result = json.loads(response.choices[0].message.content)

    tier = result.get("tier") if result.get("tier") in RISK_TIERS else "Tier 2"
    return {
        "tier": tier,
        "rationale": result.get("rationale", ""),
        "cited_doc_id": result.get("cited_doc_id"),
    }


def compliance_risk_agent(state: ComplianceState) -> dict:
    claims = state.get("claims", [])
    grounding_results = state.get("grounding_results", {})

    risk_assessments: dict[str, dict] = {}
    for claim in claims:
        evidence_chunks = grounding_results.get(claim["claim_id"], {}).get("evidence", [])
        risk_assessments[claim["claim_id"]] = _assess(claim, evidence_chunks)

    return {"risk_assessments": risk_assessments}


if __name__ == "__main__":
    sample_claims = [{"claim_id": "test1", "text": "beat the market every year", "claim_type": "performance"}]
    sample_grounding = {
        "test1": {
            "is_grounded": True,
            "evidence": [
                {
                    "doc_id": "NWP-MKT-03",
                    "section_title": "Prohibited Language List",
                    "content": '"Beat the market every year" or similar absolute performance claims '
                    "are banned in any advertisement without further compliance review and CCO sign-off.",
                }
            ],
        }
    }
    result = compliance_risk_agent({"claims": sample_claims, "grounding_results": sample_grounding})
    print(json.dumps(result, indent=2))
