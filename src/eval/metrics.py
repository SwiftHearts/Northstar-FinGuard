"""Scoring functions for the eval harness. Pure functions, no I/O or API calls, so they're
cheap to unit test independently of the (expensive) agent calls in run_eval.py.
"""

import re

TIER_ORDER = ["Tier 1", "Tier 2", "Tier 3", "Tier 4"]


def normalize_quote(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text)


def quotes_match(expected: str, predicted: str) -> bool:
    """Tolerant match: normalized quotes match if either contains the other.
    LLM-extracted spans rarely have identical boundaries (trailing punctuation,
    leading capitalization) to hand-labeled ground truth, so exact equality is
    too strict for a meaningful metric here.
    """
    a, b = normalize_quote(expected), normalize_quote(predicted)
    if not a or not b:
        return False
    return a in b or b in a


def score_claim_extraction(expected_claims: list[dict], predicted_claims: list[dict]) -> dict:
    """Greedy one-to-one matching between expected and predicted claims by quote text.
    Returns precision/recall/F1 plus the per-expected-claim match detail (used by
    downstream metrics to know which predicted claim, if any, corresponds to each
    ground-truth claim).
    """
    unmatched_predicted = list(predicted_claims)
    matches = []  # list of (expected_claim, predicted_claim_or_None)

    for expected in expected_claims:
        found = None
        for predicted in unmatched_predicted:
            if quotes_match(expected["quote"], predicted["text"]):
                found = predicted
                break
        if found is not None:
            unmatched_predicted.remove(found)
        matches.append((expected, found))

    true_positives = sum(1 for _, p in matches if p is not None)
    precision = true_positives / len(predicted_claims) if predicted_claims else 1.0
    recall = true_positives / len(expected_claims) if expected_claims else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matches,
        "extra_predicted": len(unmatched_predicted),
    }


def score_claim_type(expected: dict, predicted_claim: dict | None) -> bool | None:
    """None if there's no predicted claim to compare against (extraction miss)."""
    if predicted_claim is None:
        return None
    acceptable = expected.get("acceptable_claim_types") or [expected.get("claim_type")]
    return predicted_claim["claim_type"] in acceptable


def score_dedup(expected: dict, expected_claims: list[dict], predicted_claims: list[dict]) -> bool | None:
    """For claims flagged dedup_check=True: verify exactly one predicted claim matches
    this quote span, not two (the double-extraction bug this regression case targets).
    """
    if not expected.get("dedup_check"):
        return None
    count = sum(1 for p in predicted_claims if quotes_match(expected["quote"], p["text"]))
    return count == 1


def score_intake(expected_channel: str, expected_audience: str, predicted_metadata: dict) -> dict:
    return {
        "channel_correct": predicted_metadata.get("channel") == expected_channel,
        "audience_correct": predicted_metadata.get("audience") == expected_audience,
    }


def score_retrieval_recall(expected_doc_id: str | None, evidence_chunks: list[dict]) -> bool | None:
    """None when no specific doc is expected (clean/general_factual claims) -- not scored."""
    if expected_doc_id is None:
        return None
    return any(chunk["doc_id"] == expected_doc_id for chunk in evidence_chunks)


def score_grounding_recall(expected_doc_id: str | None, reranked_chunks: list[dict]) -> bool | None:
    if expected_doc_id is None:
        return None
    return any(chunk["doc_id"] == expected_doc_id for chunk in reranked_chunks)


def score_tier(expected_tier: str | None, predicted_tier: str | None) -> dict:
    if expected_tier is None or predicted_tier is None:
        return {"exact": None, "within_one": None}
    exact = expected_tier == predicted_tier
    within_one = abs(TIER_ORDER.index(expected_tier) - TIER_ORDER.index(predicted_tier)) <= 1
    return {"exact": exact, "within_one": within_one}


def score_faithfulness(cited_doc_id: str | None, evidence_chunks: list[dict]) -> bool:
    """True (faithful) if there's no citation, or the citation names a doc that was actually
    shown to the LLM. False means the model cited a doc_id it wasn't given -- a hallucination.
    """
    if cited_doc_id is None:
        return True
    return cited_doc_id in {chunk["doc_id"] for chunk in evidence_chunks}


def rate(values: list[bool | None]) -> float | None:
    """Mean of the non-None booleans in a list, or None if there's nothing to score."""
    scored = [v for v in values if v is not None]
    if not scored:
        return None
    return sum(scored) / len(scored)
