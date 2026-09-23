# Ethics, Limitations, and Cost-to-Scale Analysis

*Northstar-FinGuard*
*Numbers sourced from `eval_dataset/results_final.json` (12 hand-labeled cases, 14 ground-truth claims) unless otherwise noted. Pricing sourced via live web search, August 2026 — verify against the Azure pricing calculator before citing in a final submission, as these change frequently.*

## 1. Intended use and human oversight

Northstar-FinGuard is a **triage tool**, not an autonomous compliance decision-maker. It reads a marketing draft, extracts claims, retrieves supporting/contradicting policy evidence, and assigns each claim a risk tier with a rationale and citations. `review_routing_agent.py` is deliberately rule-based (no LLM call — the docstring notes none is needed) and outputs a routing recommendation based on tier.

Important scope note for this section: **the codebase does not currently implement or enforce a human sign-off gate.** It produces a recommendation; nothing in the pipeline blocks a document from being treated as "approved" without a person reviewing it. That is an organizational control that must be layered on top before any real deployment, not a feature of the code today. For a regulated-industry use case (financial marketing claims), we recommend that be a hard requirement:

- Tier 3–4 (higher risk) outputs should require named human sign-off before a claim can ship, logged and auditable.
- Tier 1–2 outputs can reasonably auto-clear for low-stakes channels, but should be spot-audited on a sample basis, not left permanently unreviewed — see §2 on why full automation isn't trustworthy enough yet.
- The system's outputs (tier, rationale, citations) should be treated as a first draft for a compliance reviewer, not a final ruling, until measured accuracy on a much larger, real-world eval set justifies more autonomy.

## 2. Known limitations (measured, not hypothetical)

These come directly from the eval harness (`src/eval/run_eval.py`), not general RAG caveats:

| Metric | Score | What it means |
|---|---|---|
| Intake audience accuracy | **66.7%** | Retail vs. mixed vs. institutional audience is frequently underdetermined from text alone — see below. |
| Claim extraction precision | **83.3%** | ~1 in 6 extracted "claims" is a false positive (over-extraction), which could waste reviewer time but is a safer failure mode than missing a real claim. |
| Grounding recall@3 | **81.8%** | After reranking to the top 3 evidence chunks, roughly 1 in 5 claims loses a relevant grounding passage that was present in the wider retrieval set. |
| Compliance tier exact accuracy | **85.7%** | 1 of 14 claims was a boundary judgment call between adjacent tiers, not a clean model error (documented in `eval_dataset/cases.json`). |
| Overall document-tier accuracy | **91.7%** | End-to-end, tier-within-one accuracy is 100% — the model doesn't miss a tier by more than one step in this eval set. |

**Audience/channel underdetermination is a fairness-adjacent design issue worth naming explicitly, not just a metric to improve.** A generic public social post genuinely does not contain the signal needed to classify its audience as retail vs. mixed — the ground truth itself sometimes reflects an assumption the drafter made, not something recoverable from the text. Chasing 100% automated accuracy on an underdetermined task would mean overfitting to spurious textual cues. The better fix is a **human-confirmable field**: let the system propose an audience/channel classification but require the submitter to confirm or correct it at intake, rather than treating the model's guess as ground truth. This also has a fairness dimension — a wrong audience classification changes which compliance tier a claim lands in, so an unconfirmed guess could systematically under- or over-flag certain claim types.

**Other limitations:**
- The 12-case eval set is small and hand-labeled by one person (project author). It covers all 8 claim types and all 4 tiers, but is not large enough to detect subtler failure modes or measure calibration across a broader claim population.
- The cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is explicitly *not* used as a pass/fail gate in `grounding_verification_agent.py`, because its similarity scores were found to be poorly calibrated for this domain during development (a verbatim match scored +8.5 vs. a genuinely relevant but differently-worded passage at −9.98). Final grounding judgment is deferred to the compliance-risk LLM call instead — meaning grounding quality is currently bottlenecked on LLM judgment, not a separately verifiable score.
- Faithfulness (no hallucinated citations) is 100% in this eval set, which is reassuring but again only 14 claims' worth of evidence.

## 3. Cost-to-scale analysis

### Pipeline call structure (from `src/agents/`, confirmed against `src/eval/run_eval.py`)

| Step | Calls | Model / service |
|---|---|---|
| Intake | 1 chat call / doc | `gpt-5.4-mini` (Azure OpenAI, temp=0, JSON mode) |
| Claim extraction | 1 chat call / doc | `gpt-5.4-mini` |
| Evidence retrieval | 1 embedding call / claim + 1 vector search / claim (TOP_K=25) | `text-embedding-3-small` + Azure AI Search |
| Grounding rerank | 0 API calls (local model) | local cross-encoder, no marginal cost |
| Compliance risk scoring | 1 chat call / claim | `gpt-5.4-mini` |
| Executive summary | 1 chat call / doc | `gpt-5.4-mini` |

So per document: **3 fixed chat calls + 1 chat call per claim**, plus **1 embedding call and 1 search query per claim**. The eval set averages 1.17 claims/doc (14 claims / 12 cases) — that ratio is the scaling variable; a document with many claims costs proportionally more, but the scaling is linear, not exponential.

### Estimated $ cost per document

The codebase does not currently log token usage (no `usage`/`token_count` fields found in `src/`), so the figures below use **engineering estimates** based on actual prompt/context sizes in `src/agents/*.py`, not measured data. Before citing a final number in the capstone report, pull real per-call token counts from LangSmith (tracing is already enabled via `LANGSMITH_TRACING=true`) — it has been recording every call this eval run made.

| Call type | Est. input tokens | Est. output tokens | Calls/doc (avg) |
|---|---|---|---|
| Intake | 600 | 150 | 1 |
| Claim extraction | 700 | 300 | 1 |
| Compliance risk (per claim) | 1,400 | 250 | 1.17 |
| Executive summary | 900 | 400 | 1 |
| Embedding (per claim) | 40 | — | 1.17 |

Pricing used (Azure OpenAI, Global Standard, as found August 2026 — **reverify before publishing**): `gpt-5-mini`-tier ≈ $0.25 / 1M input tokens, $2.00 / 1M output tokens ([Azure OpenAI pricing](https://azure.microsoft.com/en-us/pricing/details/azure-openai/)); `text-embedding-3-small` ≈ $0.02 / 1M input tokens.

- Input tokens/doc ≈ 2,200 (fixed) + 1,634 (risk calls) + 47 (embeddings) ≈ **3,880**
- Output tokens/doc ≈ 850 (fixed) + 292 (risk calls) ≈ **1,140**
- **Chat cost/doc** ≈ (3,880 × $0.25 + 1,140 × $2.00) / 1,000,000 ≈ **$0.0033**
- **Embedding cost/doc** ≈ negligible (< $0.000001)
- **≈ $0.003–0.004 per document reviewed**, essentially all of it LLM inference, none of it meaningfully from embeddings.

### The actual cost driver at scale isn't inference — it's the Search tier floor

Azure AI Search bills by provisioned tier/capacity, not per-query, so it's a **fixed monthly floor cost independent of document volume** until you outgrow the tier's storage or throughput:

- Basic tier: **~$75/month** (15 GB/partition) — this comfortably covers the current 61-chunk, 7-document knowledge base for a long time.
- Standard S1: **~$250/month** — needed once the knowledge base or query throughput grows substantially, not because of review volume itself.

Putting it together at illustrative volumes:

| Docs reviewed/month | Est. LLM inference cost | Azure AI Search (Basic) | Total |
|---|---|---|---|
| 100 | ~$0.33 | $75 | ~$75.33 |
| 1,000 | ~$3.30 | $75 | ~$78.30 |
| 10,000 | ~$33 | $75 (likely need Standard soon) | ~$283 |

**The non-obvious takeaway:** at this knowledge-base size, scaling review *volume* is nearly free on the model side — going from 100 to 10,000 documents/month moves inference cost by tens of dollars, not thousands. The real cost-to-scale question isn't "how many documents can we afford to review," it's "how big does the knowledge base get, and what query throughput does that demand" — because that's what determines the Search tier, and the Search tier is the dominant fixed cost by an order of magnitude at low-to-moderate volume. This also means the system doesn't get meaningfully cheaper per-document at scale (no real economies of scale in the LLM cost), but it doesn't get punishingly expensive either — cost scales roughly linearly with claims extracted, which scales with document volume and complexity, not with some worse-than-linear factor.

Not included above: LangSmith tracing costs (usage-based beyond its free tier — not itemized here) and human reviewer time, which for a Tier 3/4 sign-off requirement (§1) is likely the actual dominant cost at any real production volume, not the API bill.

## 4. Recommendations before production use

1. Implement an actual human sign-off gate for Tier 3–4 outputs — today `review_routing_agent.py` recommends routing but nothing enforces review.
2. Replace the model-guessed audience/channel field with a human-confirmable one at intake (§2), given it's measurably underdetermined (66.7% accuracy) rather than a solvable prompt problem.
3. Instrument real token usage (LangSmith already traces every call) and replace the estimates in §3 with measured figures before quoting a final cost number.
4. Expand the eval set well beyond 12 cases before treating any accuracy number here as representative — these are directionally useful, not statistically robust.
