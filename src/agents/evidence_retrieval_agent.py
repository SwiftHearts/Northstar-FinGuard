"""Evidence retrieval agent: for each extracted claim, runs a hybrid (vector + keyword) search
against the northstar-finguard-index to pull the policy passages most relevant to that claim.

Usage (standalone test):
    python -m src.agents.evidence_retrieval_agent
"""

import json

from azure.search.documents.models import VectorizedQuery

from src.graph.state import ComplianceState
from src.utils.azure_clients import get_azure_openai_client, get_search_client
from src.utils.config import settings

TOP_K = 25


def _embed(text: str) -> list[float]:
    client = get_azure_openai_client()
    response = client.embeddings.create(model=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT, input=[text])
    return response.data[0].embedding


def _search(claim_text: str) -> list[dict]:
    search_client = get_search_client()
    vector_query = VectorizedQuery(vector=_embed(claim_text), k_nearest_neighbors=TOP_K, fields="content_vector")

    results = search_client.search(
        search_text=claim_text,
        vector_queries=[vector_query],
        select=["id", "content", "doc_id", "doc_title", "section_title", "chunk_index"],
        top=TOP_K,
    )

    return [
        {
            "chunk_id": r["id"],
            "content": r["content"],
            "doc_id": r["doc_id"],
            "doc_title": r["doc_title"],
            "section_title": r["section_title"],
            "score": r["@search.score"],
        }
        for r in results
    ]


def evidence_retrieval_agent(state: ComplianceState) -> dict:
    claims = state.get("claims", [])
    evidence: dict[str, list[dict]] = {}

    for claim in claims:
        evidence[claim["claim_id"]] = _search(claim["text"])

    return {"evidence": evidence}


if __name__ == "__main__":
    sample_claims = [
        {"claim_id": "test1", "text": "free portfolio management", "claim_type": "fee_or_pricing"},
        {"claim_id": "test2", "text": "we're the #1 rated advisor in the region", "claim_type": "superlative"},
    ]
    result = evidence_retrieval_agent({"source_text": "", "claims": sample_claims})
    print(json.dumps(result, indent=2))
