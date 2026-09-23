"""Claim extraction agent: pulls discrete, individually-reviewable compliance claims out of
a marketing/client-communication draft (performance claims, guarantees, comparisons, etc.)
so each can be checked against the knowledge base independently.

Usage (standalone test):
    python -m src.agents.claim_extraction_agent
"""

import json
import uuid
from typing import get_args

from src.graph.state import Claim, ClaimType, ComplianceState
from src.utils.azure_clients import get_azure_openai_client
from src.utils.config import settings

VALID_CLAIM_TYPES = get_args(ClaimType)

EXTRACTION_PROMPT = """You are a compliance analyst for a registered investment adviser, reviewing a \
marketing or client-communication draft for claims that require substantiation or disclosure under \
SEC Marketing Rule 206(4)-1 and FINRA Rule 2210.

Extract every discrete claim that could require compliance review, including:
- performance claims (returns, "beat the market", track record)
- guarantees or "risk-free" / "can't lose" language
- comparative claims ("better than X")
- unqualified superlatives ("best", "#1", "top")
- testimonials or endorsements
- statements about fees or pricing ("free")
- anything implying regulatory approval or endorsement
- other material factual claims about the firm or its services

For each claim, quote the exact sentence or phrase verbatim from the draft.

Return strict JSON: {{"claims": [{{"quote": "...", "claim_type": "..."}}]}}
claim_type must be one of: {claim_types}

Draft:
---
{text}
---
"""


def _extract(text: str) -> list[dict]:
    client = get_azure_openai_client()
    response = client.chat.completions.create(
        model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=[
            {
                "role": "user",
                "content": EXTRACTION_PROMPT.format(text=text, claim_types=list(VALID_CLAIM_TYPES)),
            }
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    result = json.loads(response.choices[0].message.content)
    return result.get("claims", [])


def claim_extraction_agent(state: ComplianceState) -> dict:
    # Extract the source text in the state
    text = state["source_text"]
    
    # Use the _extract function to get raw claims from the text from the Azure OpenAI model (LLM)
    raw_claims = _extract(text)

    # List to store the validated claims and their accepted spans in the text
    claims: list[Claim] = []

    # List to track the spans of text that have already been accepted as claims, to avoid overlaps
    # Identified by the start and end character indices of the quote in the source text
    accepted_spans: list[tuple[int, int]] = []

    # Iterate over the raw claims returned by the LLM and validate them against the source text
    for raw in raw_claims:

        # Get the quote from the raw claim and strip whitespace, and return if it's empty or not present in the source text
        quote = (raw.get("quote") or "").strip()

        # If the quote is empty or not found in the source text, skip this claim
        if not quote or quote not in text:
            continue  # Drop hallucinated quotes not present in the draft

        # Find the start index of the quote
        start = text.find(quote)

        # Calculate the end index of the quote
        end = start + len(quote)

        # Check if the span of this quote overlaps with any previously accepted spans
        if any(start < a_end and end > a_start for a_start, a_end in accepted_spans):
            continue  
        accepted_spans.append((start, end))

        # Get the claim type from the raw claim, and default to "general_factual" if it's not valid
        claim_type = raw.get("claim_type")
        if claim_type not in VALID_CLAIM_TYPES:
            claim_type = "general_factual"

        # Create a dictionary representing the claim: randomly generated unique ID (uuid.uuid4()), 
        # remove the hyphens to form a 32-character hexidecimal string (.hex), slice the string to take 
        # only the first 8 characters ([:8]), the quote, and the claim type, and append it to the claims list
        claims.append({"claim_id": uuid.uuid4().hex[:8], "text": quote, "claim_type": claim_type})

    # Update the 'claims' field in the state with the extracted claims
    return {"claims": claims}


if __name__ == "__main__":
    sample = (
        "Northstar Wealth Partners is proud to say our growth portfolio beat the market "
        "every year for the last decade. Sign up today for free portfolio management, "
        "and see why we're the #1 rated advisor in the region."
    )
    result = claim_extraction_agent({"source_text": sample})
    print(json.dumps(result, indent=2))
