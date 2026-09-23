# Northstar-FinGuard

Automated compliance-review triage for a registered investment adviser's marketing and client-communication drafts. Takes a raw draft (email, social post, brochure, pitch deck, webinar copy), extracts its factual claims, checks each against a knowledge base of firm policy documents, assigns a risk severity tier with citations, and routes it to the right reviewer with a CCO-ready executive summary.

**This is a triage tool, not an autonomous compliance decision-maker.** It produces a recommendation; a human sign-off gate for higher-risk output is not yet enforced in code. See [`docs/ethics_and_cost_to_scale.md`](docs/ethics_and_cost_to_scale.md) §1 for the full scope note before any real deployment.

## Architecture

A linear 7-agent [LangGraph](https://github.com/langchain-ai/langgraph) pipeline (`src/graph/build_graph.py`), all agents sharing one typed state object (`ComplianceState`, `src/graph/state.py`):

```
intake → claim_extraction → evidence_retrieval → grounding_verification
       → compliance_risk → review_routing → executive_summary
```

| Agent | Role | LLM call? |
|---|---|---|
| `intake` | Classifies channel and audience | Yes |
| `claim_extraction` | Extracts discrete, individually-reviewable claims | Yes |
| `evidence_retrieval` | Hybrid vector + keyword search per claim against the policy knowledge base | No (embedding + search API) |
| `grounding_verification` | Local cross-encoder reranks retrieved evidence | No (local model) |
| `compliance_risk` | Assigns a Tier 1–4 severity per claim with rationale and citation | Yes |
| `review_routing` | Deterministic rule-based tier → destination/SLA lookup | No |
| `executive_summary` | Drafts the final CCO-facing summary | Yes |

**Stack:** Azure OpenAI (chat + `text-embedding-3-small`), Azure AI Search (hybrid vector index), a local `sentence-transformers` cross-encoder for reranking, LangGraph for orchestration, optional LangSmith tracing.

**Knowledge base:** 7 synthetic Northstar policy documents in [`knowledge_base/`](knowledge_base/) (investment policy, risk disclosure, marketing compliance, fee/compensation, fiduciary standards, client communications, regulatory review playbook), chunked and embedded into an Azure AI Search index by [`src/ingestion/ingest_policies.py`](src/ingestion/ingest_policies.py).

For a deeper architecture and design-decision writeup, see [`docs/system_overview_and_eval.md`](docs/system_overview_and_eval.md).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env` (or create one) with the required Azure credentials:

```
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_API_VERSION=
AZURE_OPENAI_CHAT_DEPLOYMENT=
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=
SEARCH_ENDPOINT=
SEARCH_KEY=
SEARCH_INDEX_NAME=

# optional
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=
LANGSMITH_TRACING=false
```

Then build the search index once (uploads the knowledge base):

```bash
python -m src.ingestion.ingest_policies
```

## Usage

**Streamlit UI** — paste a draft, run the review, see claims highlighted by risk tier with evidence and rationale:

```bash
streamlit run src/ui/app.py
```
Opens at `http://localhost:8501`.

**Run the pipeline directly:**

```python
from src.graph.build_graph import build_graph

graph = build_graph()
result = graph.invoke({"source_text": "your draft text..."})
```

**Run any agent standalone** (each has a `__main__` block with a sample input), e.g.:

```bash
python -m src.agents.claim_extraction_agent
```

## Evaluation

A 12-case hand-labeled eval harness (`eval_dataset/cases.json`) covering all 8 claim types and all 4 risk tiers, scored both component-by-component (each agent fed ground truth in isolation) and end-to-end:

```bash
python -m src.eval.run_eval
python -m src.eval.run_eval --save results.json   # also saves raw per-case results under eval_dataset/
```

Current results (`eval_dataset/results_final.json`): 100% claim extraction recall, 83.3% precision, 100% evidence retrieval recall@5, 85.7% compliance-tier exact accuracy, 91.7% end-to-end document-tier accuracy. Full metric history and known limitations are in [`docs/system_overview_and_eval.md`](docs/system_overview_and_eval.md) and [`docs/ethics_and_cost_to_scale.md`](docs/ethics_and_cost_to_scale.md).

## Docs

- [`docs/ethics_and_cost_to_scale.md`](docs/ethics_and_cost_to_scale.md) — human oversight, measured limitations, cost-to-scale analysis
- [`docs/system_overview_and_eval.md`](docs/system_overview_and_eval.md) — architecture detail, full eval results, design decisions and tradeoffs
