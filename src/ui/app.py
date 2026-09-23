"""Streamlit UI for Northstar-FinGuard: paste a draft marketing/client communication, run it
through the compliance-review pipeline, and see per-claim risk highlighting plus the supporting
policy citations behind each finding.

Usage:
    streamlit run src/ui/app.py
"""

import html
import sys
from pathlib import Path

import streamlit as st

# `streamlit run src/ui/app.py` only puts src/ui/ on sys.path, so add the repo root to make
# the `src` package importable (needed on Streamlit Community Cloud).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.graph.build_graph import build_graph  # noqa: E402

st.set_page_config(page_title="Northstar-FinGuard", layout="wide")

TIER_COLORS = {
    "Tier 1": "#1f9d55",
    "Tier 2": "#d4a017",
    "Tier 3": "#d9534f",
    "Tier 4": "#8b0000",
}


@st.cache_resource
def get_app():
    return build_graph()


def render_highlighted_draft(source_text: str, claims: list[dict], risk_assessments: dict) -> str:
    spans = []
    for claim in claims:
        start = source_text.find(claim["text"])
        if start == -1:
            continue
        tier = risk_assessments.get(claim["claim_id"], {}).get("tier", "Tier 1")
        spans.append((start, start + len(claim["text"]), tier))

    spans.sort(key=lambda s: s[0])

    parts = []
    cursor = 0
    for start, end, tier in spans:
        if start < cursor:
            continue  # overlapping claim spans; keep the earlier one
        parts.append(html.escape(source_text[cursor:start]))
        color = TIER_COLORS.get(tier, "#1f9d55")
        parts.append(
            f'<mark style="background-color:{color}33;border-bottom:2px solid {color};'
            f'padding:1px 2px;">{html.escape(source_text[start:end])}</mark>'
        )
        cursor = end
    parts.append(html.escape(source_text[cursor:]))

    return "".join(parts)


st.title("Northstar-FinGuard")
st.caption("Automated compliance review for marketing and client-communication drafts")

draft_text = st.text_area(
    "Paste a draft communication",
    height=180,
    placeholder="e.g. Northstar Wealth Partners is proud to say our growth portfolio beat the "
    "market every year for the last decade...",
)

run_clicked = st.button("Run compliance review", type="primary", disabled=not draft_text.strip())

if run_clicked:
    with st.spinner("Running intake, claim extraction, evidence retrieval, and risk assessment..."):
        app = get_app()
        st.session_state["result"] = app.invoke({"source_text": draft_text})

result = st.session_state.get("result")

if result:
    claims = result.get("claims", [])
    risk_assessments = result.get("risk_assessments", {})
    review_tier = result.get("review_tier", "Tier 1")

    st.subheader("Overall assessment")
    tier_color = TIER_COLORS.get(review_tier, "#1f9d55")
    st.markdown(
        f'<span style="background-color:{tier_color};color:white;padding:4px 10px;'
        f'border-radius:4px;font-weight:600;">{review_tier}</span>',
        unsafe_allow_html=True,
    )
    st.write(result.get("routing_notes", ""))

    st.subheader("Draft (highlighted by claim risk)")
    st.markdown(
        render_highlighted_draft(result.get("source_text", ""), claims, risk_assessments),
        unsafe_allow_html=True,
    )
    legend = "&nbsp;&nbsp;".join(
        f'<span style="background-color:{color}33;border-bottom:2px solid {color};'
        f'padding:0 4px;">{tier}</span>'
        for tier, color in TIER_COLORS.items()
    )
    st.markdown(legend, unsafe_allow_html=True)

    st.subheader("Executive summary")
    st.write(result.get("executive_summary", ""))

    st.subheader("Claim-by-claim findings")
    for claim in claims:
        assessment = risk_assessments.get(claim["claim_id"], {})
        tier = assessment.get("tier", "unassessed")
        with st.expander(f'{tier} — "{claim["text"]}"'):
            st.markdown(f"**Claim type:** {claim['claim_type']}")
            st.markdown(f"**Rationale:** {assessment.get('rationale', '')}")
            st.markdown(f"**Cited policy:** {assessment.get('cited_doc_id') or 'None'}")

            evidence_chunks = result.get("grounding_results", {}).get(claim["claim_id"], {}).get("evidence", [])
            if evidence_chunks:
                st.markdown("**Supporting evidence retrieved:**")
                for chunk in evidence_chunks:
                    st.markdown(f"- *{chunk['doc_id']} — {chunk['section_title']}*")
                    preview = chunk["content"][:300] + ("..." if len(chunk["content"]) > 300 else "")
                    st.caption(preview)

    with st.expander("Draft metadata"):
        st.json(result.get("metadata", {}))
