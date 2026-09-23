"""Eval harness for the Northstar-FinGuard compliance pipeline.

Runs two kinds of checks against eval_dataset/cases.json:

1. Component-level: each agent is run in isolation against hand-labeled ground truth
   (e.g. evidence_retrieval_agent is fed the *ground-truth* claims, not whatever
   claim_extraction_agent happened to produce) so a failure in one stage doesn't mask
   or get confused with a failure in another. This is what makes retrieval recall a
   clean, actionable number instead of "the pipeline was wrong somewhere."
2. End-to-end: review_routing_agent's document-level tier is checked against each
   case's expected_overall_tier, using the ground-truth-derived risk assessments.

Usage:
    python -m src.eval.run_eval
    python -m src.eval.run_eval --save results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

from src.agents.claim_extraction_agent import claim_extraction_agent
from src.agents.compliance_risk_agent import compliance_risk_agent
from src.agents.evidence_retrieval_agent import evidence_retrieval_agent
from src.agents.grounding_verification_agent import grounding_verification_agent
from src.agents.intake_agent import intake_agent
from src.agents.review_routing_agent import review_routing_agent
from src.eval import metrics
from src.utils.config import settings

CASES_PATH = Path(__file__).resolve().parents[2] / "eval_dataset" / "cases.json"


def _ground_truth_claim_type(expected: dict) -> str:
    if "claim_type" in expected:
        return expected["claim_type"]
    return expected["acceptable_claim_types"][0]


def run_case(case: dict) -> dict:
    result: dict = {"case_id": case["case_id"]}

    # --- Intake ---
    intake_out = intake_agent({"source_text": case["source_text"]})
    result["intake"] = metrics.score_intake(
        case["expected_channel"], case["expected_audience"], intake_out["metadata"]
    )

    # --- Claim extraction (against the raw draft, scored against ground truth) ---
    extraction_out = claim_extraction_agent({"source_text": case["source_text"]})
    predicted_claims = extraction_out["claims"]
    extraction_score = metrics.score_claim_extraction(case["claims"], predicted_claims)
    result["claim_extraction"] = {
        "precision": extraction_score["precision"],
        "recall": extraction_score["recall"],
        "f1": extraction_score["f1"],
        "extra_predicted": extraction_score["extra_predicted"],
    }
    result["claim_type_correct"] = [
        metrics.score_claim_type(expected, predicted) for expected, predicted in extraction_score["matches"]
    ]
    result["dedup_checks"] = [
        d
        for d in (metrics.score_dedup(expected, case["claims"], predicted_claims) for expected in case["claims"])
        if d is not None
    ]

    # --- Ground-truth-driven claims for isolated retrieval/grounding/risk testing ---
    gt_claims = [
        {"claim_id": c["claim_id"], "text": c["quote"], "claim_type": _ground_truth_claim_type(c)}
        for c in case["claims"]
    ]

    retrieval_out = evidence_retrieval_agent({"claims": gt_claims})
    result["retrieval_recall"] = [
        metrics.score_retrieval_recall(c.get("expected_doc_id"), retrieval_out["evidence"].get(c["claim_id"], []))
        for c in case["claims"]
    ]

    grounding_out = grounding_verification_agent({"claims": gt_claims, "evidence": retrieval_out["evidence"]})
    result["grounding_recall"] = [
        metrics.score_grounding_recall(
            c.get("expected_doc_id"), grounding_out["grounding_results"].get(c["claim_id"], {}).get("evidence", [])
        )
        for c in case["claims"]
    ]

    risk_out = compliance_risk_agent({"claims": gt_claims, "grounding_results": grounding_out["grounding_results"]})
    result["tier_scores"] = [
        metrics.score_tier(c.get("expected_tier"), risk_out["risk_assessments"].get(c["claim_id"], {}).get("tier"))
        for c in case["claims"]
    ]
    result["faithfulness"] = [
        metrics.score_faithfulness(
            risk_out["risk_assessments"].get(c["claim_id"], {}).get("cited_doc_id"),
            grounding_out["grounding_results"].get(c["claim_id"], {}).get("evidence", []),
        )
        for c in case["claims"]
    ]

    routing_out = review_routing_agent({"risk_assessments": risk_out["risk_assessments"]})
    result["overall_tier"] = {
        "expected": case["expected_overall_tier"],
        "actual": routing_out["review_tier"],
        "correct": routing_out["review_tier"] == case["expected_overall_tier"],
    }

    return result


def aggregate(results: list[dict]) -> dict:
    def flat(key_path):
        out = []
        for r in results:
            val = r
            for key in key_path:
                val = val[key]
            if isinstance(val, list):
                out.extend(val)
            else:
                out.append(val)
        return out

    micro_precisions = [r["claim_extraction"]["precision"] for r in results]
    micro_recalls = [r["claim_extraction"]["recall"] for r in results]

    return {
        "n_cases": len(results),
        "claim_extraction_precision_avg": sum(micro_precisions) / len(micro_precisions),
        "claim_extraction_recall_avg": sum(micro_recalls) / len(micro_recalls),
        "claim_type_accuracy": metrics.rate(flat(["claim_type_correct"])),
        "dedup_pass_rate": metrics.rate(flat(["dedup_checks"])),
        "intake_channel_accuracy": metrics.rate([r["intake"]["channel_correct"] for r in results]),
        "intake_audience_accuracy": metrics.rate([r["intake"]["audience_correct"] for r in results]),
        "retrieval_recall_at_5": metrics.rate(flat(["retrieval_recall"])),
        "grounding_recall_at_3": metrics.rate(flat(["grounding_recall"])),
        "tier_exact_accuracy": metrics.rate([s["exact"] for r in results for s in r["tier_scores"]]),
        "tier_within_one_accuracy": metrics.rate([s["within_one"] for r in results for s in r["tier_scores"]]),
        "faithfulness_rate": metrics.rate(flat(["faithfulness"])),
        "overall_tier_accuracy": metrics.rate([r["overall_tier"]["correct"] for r in results]),
    }


def fmt_pct(value):
    return "n/a" if value is None else f"{value * 100:.0f}%"


def print_report(results: list[dict], summary: dict) -> None:
    print("\n=== Northstar-FinGuard Eval Report ===\n")
    print(f"Cases run: {summary['n_cases']}")
    print(f"Claim extraction   precision/recall: {fmt_pct(summary['claim_extraction_precision_avg'])} / "
          f"{fmt_pct(summary['claim_extraction_recall_avg'])}")
    print(f"Claim type accuracy (of matched claims): {fmt_pct(summary['claim_type_accuracy'])}")
    print(f"Dedup regression pass rate:               {fmt_pct(summary['dedup_pass_rate'])}")
    print(f"Intake channel accuracy:                  {fmt_pct(summary['intake_channel_accuracy'])}")
    print(f"Intake audience accuracy:                 {fmt_pct(summary['intake_audience_accuracy'])}")
    print(f"Evidence retrieval recall@5:               {fmt_pct(summary['retrieval_recall_at_5'])}")
    print(f"Grounding (rerank) recall@3:                {fmt_pct(summary['grounding_recall_at_3'])}")
    print(f"Compliance tier exact accuracy:            {fmt_pct(summary['tier_exact_accuracy'])}")
    print(f"Compliance tier within-1 accuracy:         {fmt_pct(summary['tier_within_one_accuracy'])}")
    print(f"Citation faithfulness (no hallucinated doc_id): {fmt_pct(summary['faithfulness_rate'])}")
    print(f"Document-level (routing) tier accuracy:    {fmt_pct(summary['overall_tier_accuracy'])}")

    print("\n--- Per-case retrieval recall (flags the known superlative regression) ---")
    for r in results:
        misses = [i for i, v in enumerate(r["retrieval_recall"]) if v is False]
        flag = "  <-- MISS" if misses else ""
        print(f"  {r['case_id']:40s} recall={r['retrieval_recall']}{flag}")

    print("\n--- Cases with overall tier mismatch ---")
    any_mismatch = False
    for r in results:
        if not r["overall_tier"]["correct"]:
            any_mismatch = True
            print(f"  {r['case_id']}: expected {r['overall_tier']['expected']}, got {r['overall_tier']['actual']}")
    if not any_mismatch:
        print("  (none)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", help="Optional filename (written under eval_dataset/) for raw per-case results")
    args = parser.parse_args()

    if settings.LANGSMITH_TRACING == "true" and settings.LANGSMITH_API_KEY:
        print(f"LangSmith tracing enabled -- project '{settings.LANGSMITH_PROJECT}'")
    else:
        print("LangSmith tracing not enabled (set LANGSMITH_TRACING=true and LANGSMITH_API_KEY in .env to trace runs)")

    cases = json.loads(CASES_PATH.read_text())["cases"]

    results = []
    start = time.time()
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] running {case['case_id']}...", file=sys.stderr)
        results.append(run_case(case))
    elapsed = time.time() - start

    summary = aggregate(results)
    print_report(results, summary)
    print(f"\nElapsed: {elapsed:.1f}s")

    if args.save:
        out_path = CASES_PATH.parent / args.save
        out_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2, default=str))
        print(f"Saved raw results to {out_path}")


if __name__ == "__main__":
    main()
