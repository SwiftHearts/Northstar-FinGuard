"""Grounding verification agent: reranks each claim's retrieved evidence with a cross-encoder
so the strongest-matching passages are surfaced first for the compliance-risk step.

Note on design: ms-marco-MiniLM-L-6-v2 logits are not well-calibrated as an absolute relevance
gate for this domain -- a clearly on-topic regulatory passage can score similarly to an unrelated
one unless the wording nearly matches verbatim (verified empirically: a verbatim phrase match
scored +8.5 while a genuinely on-point but differently-worded passage scored -9.98). So this
agent uses the cross-encoder only for *relative* reordering within each claim's candidates, not
as a pass/fail cutoff. "is_grounded" here just reflects whether retrieval (Agent 3) found any
candidates at all; the actual judgment of whether the top passages truly substantiate or
contradict the claim is deferred to the compliance-risk agent's LLM call, which sees the full
passage text and can return a null citation if nothing truly applies.

Usage (standalone test):
    python -m src.agents.grounding_verification_agent
"""

import json

from sentence_transformers import CrossEncoder

from src.graph.state import ComplianceState

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_N_GROUNDED = 3

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(CROSS_ENCODER_MODEL)
    return _model


def grounding_verification_agent(state: ComplianceState) -> dict:
    claims = state.get("claims", [])
    evidence = state.get("evidence", {})
    model = _get_model()

    grounding_results: dict[str, dict] = {}

    for claim in claims:
        claim_id = claim["claim_id"]
        chunks = evidence.get(claim_id, [])

        if not chunks:
            grounding_results[claim_id] = {"is_grounded": False, "evidence": []}
            continue

        pairs = [(claim["text"], chunk["content"]) for chunk in chunks]
        rerank_scores = model.predict(pairs)

        reranked = sorted(
            ({**chunk, "relevance_score": float(score)} for chunk, score in zip(chunks, rerank_scores)),
            key=lambda c: c["relevance_score"],
            reverse=True,
        )

        top_chunks = reranked[:TOP_N_GROUNDED]

        grounding_results[claim_id] = {
            "is_grounded": len(top_chunks) > 0,
            "evidence": top_chunks,
        }

    return {"grounding_results": grounding_results}


if __name__ == "__main__":
    sample_claims = [{"claim_id": "test1", "text": "beat the market every year", "claim_type": "performance"}]
    sample_evidence = {
        "test1": [
            {
                "chunk_id": "a",
                "content": "Cherry-picking is prohibited. A firm may not selectively present only its "
                "best-performing accounts.",
                "doc_id": "NWP-MKT-03",
                "doc_title": "Marketing Compliance Guidelines",
                "section_title": "Performance Advertising Rules",
                "score": 2.1,
            },
            {
                "chunk_id": "b",
                "content": "Fee schedules are reviewed annually and billed quarterly in advance.",
                "doc_id": "NWP-FCP-04",
                "doc_title": "Fee and Compensation Policy",
                "section_title": "Fee Review",
                "score": 1.2,
            },
        ]
    }
    result = grounding_verification_agent({"claims": sample_claims, "evidence": sample_evidence})
    print(json.dumps(result, indent=2))
