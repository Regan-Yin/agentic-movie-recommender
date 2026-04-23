"""
End-to-end example of the validation methodology.

What it does:
  1. Loads the TMDB CSV and the DSPy/GEPA eval cases.
  2. Builds a ValidationFramework whose baseline is restricted to
     `llm.TOP_MOVIES` (so the baseline and the agent compete on the same pool).
  3. For each case, calls the live agent via `llm.get_recommendation(...)`
     and scores its pick against the baseline.
  4. Prints an aggregate report and saves:
       - validation_results.csv    (one row per case)
       - validation_summary.json   (aggregate metrics)

Run:
  python run_validation_example.py
  python run_validation_example.py --limit 5
  python run_validation_example.py --cases my_cases.json
  python run_validation_example.py --no-live       # skip the live agent call (use stub)

Prerequisites:
  pip install -r requirements.txt
  export OLLAMA_API_KEY=...        # otherwise agent falls through to fallback path
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from validation import ValidationFramework, build_validator  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_eval_cases(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    for i, c in enumerate(cases):
        c.setdefault("case_id", f"case_{i+1:03d}")
    return cases


def _stub_agent_from_baseline(validator: ValidationFramework):
    """
    Non-LLM fallback agent used when --no-live is passed. It returns the
    baseline top-1 — mostly useful for sanity-checking the framework itself
    without burning Ollama credits.
    """
    def call(preferences: str, history: List[str], history_ids: List[int]) -> Dict:
        recs = validator.baseline.recommend(
            preferences=preferences,
            watch_history=history,
            watch_history_ids=history_ids,
            top_k=1,
        )
        if not recs:
            return {"tmdb_id": 0, "description": ""}
        tid, title, _ = recs[0]
        return {"tmdb_id": int(tid), "description": f"{title}: baseline stub."}

    return call


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the independent validation end-to-end.")
    parser.add_argument(
        "--cases",
        default=os.path.join(PROJECT_ROOT, "dspy_gepa_eval_cases.json"),
        help="Path to eval cases JSON (default: dspy_gepa_eval_cases.json).",
    )
    parser.add_argument(
        "--data",
        default=os.path.join(PROJECT_ROOT, "tmdb_top1000_movies.csv"),
        help="Path to TMDB CSV.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit number of cases (0 = all).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed when sampling cases.")
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Randomly sample N cases from the file (0 = take them in order). "
        "Useful for a quick smoke test without hammering the LLM.",
    )
    parser.add_argument(
        "--no-live",
        action="store_true",
        help="Skip the live agent call; use the baseline-as-stub. "
        "Lets you verify the framework without spending LLM credits.",
    )
    parser.add_argument(
        "--full-pool",
        action="store_true",
        help="Let the baseline search the full CSV instead of just llm.TOP_MOVIES.",
    )
    parser.add_argument(
        "--results-csv",
        default=os.path.join(PROJECT_ROOT, "validation_results.csv"),
    )
    parser.add_argument(
        "--summary-json",
        default=os.path.join(PROJECT_ROOT, "validation_summary.json"),
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Step 1: dataset + cases
    # ------------------------------------------------------------------

    print("\n" + "=" * 75)
    print("MOVIE RECOMMENDATION AGENT — INDEPENDENT VALIDATION")
    print("=" * 75)

    if not os.path.exists(args.data):
        print(f"\n❌ Dataset not found: {args.data}")
        return
    if not os.path.exists(args.cases):
        print(f"\n❌ Cases file not found: {args.cases}")
        return

    print(f"\n📂 Loading dataset: {args.data}")
    movies_df = pd.read_csv(args.data)
    print(f"   ✓ {len(movies_df)} movies")

    print(f"\n📋 Loading eval cases: {args.cases}")
    cases = _load_eval_cases(args.cases)
    print(f"   ✓ {len(cases)} cases in file")

    if args.sample > 0 and args.sample < len(cases):
        random.seed(args.seed)
        cases = random.sample(cases, args.sample)
        print(f"   → sampled {len(cases)} (seed={args.seed})")
    if args.limit > 0:
        cases = cases[: args.limit]
        print(f"   → truncated to {len(cases)}")

    # ------------------------------------------------------------------
    # Step 2: validator
    # ------------------------------------------------------------------

    print("\n🔧 Building validator")
    validator = build_validator(
        movies_df=movies_df,
        use_agent_pool=not args.full_pool,
    )
    print(f"   ✓ Baseline pool: {len(validator.baseline.movies)} movies")
    print(f"   ✓ Soft-match threshold: {validator.SOFT_MATCH_THRESHOLD}")

    # ------------------------------------------------------------------
    # Step 3: agent callable
    # ------------------------------------------------------------------

    if args.no_live:
        print("\n⚠  Live agent disabled — using baseline-top-1 stub.")
        agent_callable = _stub_agent_from_baseline(validator)
    else:
        print("\n🤖 Importing live agent (llm.get_recommendation)")
        try:
            from llm import get_recommendation as agent_callable  # type: ignore
        except Exception as exc:
            print(f"❌ Could not import llm.get_recommendation: {exc!r}")
            return
        if not os.environ.get("OLLAMA_API_KEY"):
            print("   ⚠  OLLAMA_API_KEY not set — the agent will use its deterministic fallback path.")
        else:
            print("   ✓ OLLAMA_API_KEY detected — the agent will use Ollama.")

    # ------------------------------------------------------------------
    # Step 4: run validation
    # ------------------------------------------------------------------

    print("\n▶️  Running validation\n")
    run_start = time.perf_counter()
    successful = failed = 0

    for i, case in enumerate(cases, 1):
        cid = case["case_id"]
        prefs = case.get("preferences", "")
        history = case.get("history", []) or []
        history_ids = case.get("history_ids", []) or []

        print(f"  [{i:02d}/{len(cases)}] {cid}: {str(prefs)[:58]}")
        try:
            result = validator.validate_with_agent(
                case_id=cid,
                preferences=prefs,
                watch_history=history,
                watch_history_ids=history_ids,
                agent_callable=agent_callable,
            )
        except Exception as exc:
            print(f"        ✗ framework error: {exc!r}")
            failed += 1
            continue

        if result.get("status") == "success":
            lat = result.get("agent_latency_s")
            lat_str = f"{lat:.2f}s" if lat is not None else "-"
            sm = "SOFT" if result["soft_match"] else "    "
            em = "EXACT" if result["exact_match"] else "     "
            print(
                f"        ✓ {em} {sm}  rank={result['baseline_rank']}  "
                f"hybrid={result['hybrid_similarity']:.2f}  latency={lat_str}"
            )
            print(f"            agent:    {result['agent_title']}")
            print(f"            baseline: {result['baseline_title']}")
            successful += 1
        else:
            print(f"        ✗ {result.get('status')}: {result.get('error', '')}")
            failed += 1

    run_elapsed = time.perf_counter() - run_start

    # ------------------------------------------------------------------
    # Step 5: report + persist
    # ------------------------------------------------------------------

    validator.print_report()

    print("💾 Saving artifacts")
    try:
        validator.save_results_csv(args.results_csv)
    except Exception as exc:
        print(f"   ✗ failed to save CSV: {exc!r}")
    try:
        validator.save_summary_json(args.summary_json)
    except Exception as exc:
        print(f"   ✗ failed to save JSON: {exc!r}")

    # ------------------------------------------------------------------
    # Step 6: quick interpretation
    # ------------------------------------------------------------------

    report = validator.get_summary_report()
    metrics = report.get("metrics", {}) if isinstance(report, dict) else {}
    print("\n📊 INTERPRETATION")
    print("   ───────────────────────────────────────────────────────────")
    print(f"   Avg Rank Score :  {metrics.get('avg_rank_score', 0):.1f}  (100 = perfect alignment)")
    print(f"   Soft Match Rate:  {metrics.get('soft_match_rate', 0):.1%}  (primary practical accuracy)")
    print(f"   Top-10 Hit Rate:  {metrics.get('top10_hit_rate', 0):.1%}  (in the baseline front-runners)")
    print(f"   Hybrid Similarity:{metrics.get('avg_hybrid_similarity', 0):.3f}")
    if "avg_agent_latency_s" in metrics:
        print(f"   Avg Latency:     {metrics['avg_agent_latency_s']:.2f}s (budget: 20s)")
        print(f"   Max Latency:     {metrics['max_agent_latency_s']:.2f}s")
    print(f"   Description ≤500c:{metrics.get('description_within_limit_rate', 0):.1%}")
    print(f"   History Respected:{metrics.get('history_respected_rate', 0):.1%}")
    print("   ───────────────────────────────────────────────────────────")

    soft = metrics.get("soft_match_rate", 0.0)
    if soft >= 0.50:
        print("   ✓ Agent aligns strongly with the data-driven baseline.")
    elif soft >= 0.30:
        print("   ~ Agent shows reasonable agreement. Scan validation_results.csv for outliers.")
    else:
        print("   ⚠ Agent diverges substantially. Investigate individual cases.")

    print(f"\n✅ Done in {run_elapsed:.1f}s")
    print(f"   Tests run:  {successful + failed}")
    print(f"   Successful: {successful}")
    print(f"   Failed:     {failed}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
