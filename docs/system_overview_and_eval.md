# System Overview, Evaluation Results, and Design Decisions

*Northstar-FinGuard*
*Grounded directly in the code (`src/`) and `eval_dataset/`. Companion to `docs/ethics_and_cost_to_scale.md`, which covers oversight, limitations, and cost — not repeated here.*

## 1. System overview & architecture

Northstar-FinGuard is a compliance-review triage tool for a registered investment adviser: it takes a raw draft marketing/client communication and produces a routed, CCO-ready compliance review with per-claim risk tiers, citations, and an executive summary.

### Pipeline

Built as a LangGraph `StateGraph` (`src/graph/build_graph.py`) — a **linear** chain of seven agent nodes sharing one typed state object (`ComplianceState`, `src/graph/state.py`):

```
intake → claim_extraction → evidence_retrieval → grounding_verification
       → compliance_risk → review_routing → executive_summary
```

| Agent | Role | LLM call? |
|---|---|---|
| `intake_agent` | Classifies channel (email/social/brochure/pitch_deck/webinar/other) and audience (retail/institutional/mixed); assigns a `draft_id` | Yes (1/doc) |
| `claim_extraction_agent` | Pulls discrete, individually-reviewable claims (performance, guarantee, comparative, superlative, testimonial, fee/pricing, regulatory-endorsement, general-factual) | Yes (1/doc) |
| `evidence_retrieval_agent` | Hybrid vector + keyword search per claim against the policy knowledge base, top 25 candidates | No (embedding + search API) |
| `grounding_verification_agent` | Local cross-encoder reranks each claim's 25 candidates down to the top 3 | No (local model) |
| `compliance_risk_agent` | Assigns a severity tier (1–4) per claim with rationale and a citation, per the firm's Regulatory Review Playbook | Yes (1/claim) |
| `review_routing_agent` | Aggregates claim tiers to a document-level tier and routing destination | No — pure rule lookup |
| `executive_summary_agent` | Drafts a 4–6 sentence CCO-ready summary of the whole review | Yes (1/doc) |

State accumulates through the graph: `source_text`/`metadata` → `claims` → `evidence` → `grounding_results` → `risk_assessments` → `review_tier`/`routing_notes` → `executive_summary`.

### Azure services

- **Azure OpenAI** — chat completions (intake, extraction, risk scoring, summary — all JSON-mode where structured output is needed, `temperature=0` except the summary at `0.2`) and `text-embedding-3-small` for embeddings.
- **Azure AI Search** — hybrid vector + keyword index (`northstar-finguard-index`), HNSW vector search, holding the chunked policy knowledge base.
- **Local cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`, via `sentence-transformers`) — reranking only, no API cost or network call.
- **LangSmith** — optional tracing, gated by `LANGSMITH_TRACING` env var.

### Knowledge base & ingestion

`src/ingestion/ingest_policies.py` chunks the 7 synthetic Northstar policy documents in `knowledge_base/` (investment policy, risk disclosure, marketing compliance, fee/compensation, fiduciary standards, client communications, regulatory review playbook) by markdown section, splitting long sections at 1,500 chars with 150-char overlap, embeds each chunk, and uploads to the search index with doc/section metadata preserved for citation.

Config (`src/utils/config.py`) is entirely env-var driven with a `validate_settings()` guard that fails fast if required Azure credentials/endpoints are missing.

## 2. Evaluation results

### Methodology (`src/eval/run_eval.py`, `src/eval/metrics.py`)

12 hand-labeled cases (`eval_dataset/cases.json`), 14 ground-truth claims, covering all 8 claim types and all 4 risk tiers. Two evaluation styles run together:

1. **Component-level, isolated** — each agent downstream of extraction is fed *ground-truth* claims rather than whatever the previous agent actually produced, so a failure in one stage can't mask or get blamed on another (e.g. retrieval recall is a clean number about retrieval, not "the pipeline was wrong somewhere").
2. **End-to-end** — `review_routing_agent`'s document-level tier is checked against each case's expected overall tier.

Claim-text matching uses a tolerant substring comparison (`quotes_match`), not exact equality — LLM-extracted spans rarely hit identical boundaries to hand-labeled ground truth (trailing punctuation, capitalization).

### Results across three iterations

The repo has three saved eval runs — `results_baseline.json` → `results_after_fixes.json` → `results_final.json` — documenting iterative improvement:

| Metric | Baseline | After fixes | Final |
|---|---|---|---|
| Claim extraction precision | 68.1% | 73.6% | **83.3%** |
| Claim extraction recall | 100% | 100% | 100% |
| Claim type accuracy | 100% | 100% | 100% |
| Dedup regression pass rate | 0% | 100% | 100% |
| Intake channel accuracy | 66.7% | **100%** | **100%** |
| Intake audience accuracy | 83.3% | 83.3% | 66.7% |
| Evidence retrieval recall@5 | 45.5% | 63.6% | **100%** |
| Grounding recall@3 | 45.5% | 63.6% | 81.8% |
| Compliance tier exact accuracy | 50.0% | 50.0% | **85.7%** |
| Compliance tier within-1 accuracy | 100% | 100% | 100% |
| Citation faithfulness | 100% | 100% | 100% |
| Overall (document-level) tier accuracy | 58.3% | 58.3% | **91.7%** |

**Worth flagging, not smoothing over:** intake audience accuracy is the one metric that got *worse* over the iteration series (83.3% → 66.7%) while everything else improved — likely a side effect of whatever prompt/logic change fixed channel accuracy to 100%, or noise from the small 12-case set (each case is ~8.3 points). This is the same underdetermination issue discussed in `docs/ethics_and_cost_to_scale.md` §2 (audience often genuinely isn't recoverable from text alone) rather than a straightforward regression to chase.

**Remaining known miss in the final run:** case `clean_multi_claim_institutional` — expected `Tier 1`, got `Tier 2`. This is an over-flagging error (safer failure direction than under-flagging a real violation), and evidence retrieval recall is 100% in the final run, so it's a risk-scoring judgment call rather than a retrieval gap.

## 3. Design decisions & tradeoffs

- **Routing is rule-based, not LLM-based** (`review_routing_agent.py`). Once each claim's tier is known, mapping to a destination/SLA is a deterministic lookup — an LLM call there would add cost, latency, and a hallucination surface for zero benefit.
- **The cross-encoder reranker is used only for relative reordering, not as a pass/fail gate.** Documented empirically in the code: a verbatim phrase match scored +8.5 while a genuinely on-topic but differently-worded passage scored −9.98 on the same model — absolute thresholds would have discarded good evidence. Final grounding judgment is deferred to the compliance-risk LLM call, which sees full passage text and can return a null citation itself.
- **Claim extraction guards against two specific failure modes**: hallucinated quotes (any extracted quote not found verbatim in the source text is dropped) and overlapping spans (a span already claimed under one claim type can't be re-claimed under another) — this is what the `dedup_pass_rate` metric regression-tests (0% → 100% after the fix).
- **Determinism where it matters, variance where it's fine**: `temperature=0` for every classification/extraction/scoring call (intake, extraction, risk); `temperature=0.2` only for the executive summary, where some prose variation is acceptable and arguably desirable.
- **Component-level eval isolation was a deliberate harness design choice**, not an artifact of how the pipeline runs live — in production each agent consumes the previous agent's actual output; the eval harness re-feeds ground truth at each stage specifically so metrics attribute failure to the right component.
- **No human sign-off gate is enforced in code** — `review_routing_agent` produces a recommendation only. This is called out as a real gap (not a design tradeoff to defend) in `docs/ethics_and_cost_to_scale.md` §1 and §4.
