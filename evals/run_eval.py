"""
Evaluation runner — runs the agent over the ground truth dataset and judges each response.

Usage:
    uv run python evals/run_eval.py           # standard eval run
    uv run python evals/run_eval.py --tune    # compare chunk sizes and num_results
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Allow running from repo root or from evals/
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from ai_learning_os.agent import agent
from ai_learning_os.config import get_config
from evals.judge import evaluate

GROUND_TRUTH_FILE = Path(__file__).parent / "ground_truth.json"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_single(entry: dict, cfg, client: OpenAI) -> dict:
    """Run agent on one ground truth entry and judge the result."""
    print(f"  Running: {entry['id']} — {entry['question'][:60]}...")
    t0 = time.time()
    result = agent(entry["question"], cfg=cfg, client=client)
    latency = (time.time() - t0) * 1000

    verdict = evaluate(
        user_prompt=entry["question"],
        answer=result.answer,
        tool_calls=result.tool_calls,
        criteria=entry["criteria"],
    )

    # Check expected tools
    called_tools = [tc["name"] for tc in result.tool_calls]
    expected_tools = entry.get("expected_tools", [])
    tools_ok = all(t in called_tools for t in expected_tools)

    status = "PASS" if verdict.overall_passed and tools_ok else "FAIL"
    print(f"    {status} — {verdict.summary}")

    return {
        "id": entry["id"],
        "question": entry["question"],
        "status": status,
        "overall_passed": verdict.overall_passed,
        "tools_ok": tools_ok,
        "expected_tools": expected_tools,
        "called_tools": called_tools,
        "criteria_results": [r.model_dump() for r in verdict.criteria_results],
        "summary": verdict.summary,
        "answer": result.answer,
        "latency_ms": round(latency),
    }


def run_eval(cfg_overrides: dict | None = None) -> dict:
    cfg = get_config()
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            setattr(cfg, k, v)
    client = OpenAI()

    ground_truth = json.loads(GROUND_TRUTH_FILE.read_text())
    print(f"\nRunning eval on {len(ground_truth)} ground truth entries...")
    print(f"Config: model={cfg.llm_model}, chunk_size={cfg.chunk_size_words}, num_results={cfg.search_num_results}\n")

    results = []
    for entry in ground_truth:
        results.append(run_single(entry, cfg, client))

    passed = sum(1 for r in results if r["status"] == "PASS")
    total = len(results)
    pass_rate = passed / total * 100 if total else 0

    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "config": {
            "llm_model": cfg.llm_model,
            "chunk_size_words": cfg.chunk_size_words,
            "chunk_overlap_words": cfg.chunk_overlap_words,
            "search_num_results": cfg.search_num_results,
        },
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate_pct": round(pass_rate, 1),
        "results": results,
    }

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_file = RESULTS_DIR / f"run_{ts}.json"
    out_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n{'='*60}")
    print(f"RESULT: {passed}/{total} passed ({pass_rate:.1f}%)")
    print(f"Saved to: {out_file}")
    print("="*60)

    return summary


def run_tuning():
    """Compare different chunk sizes and num_results — for retrieval tuning."""
    configs = [
        {"chunk_size_words": 200, "search_num_results": 3, "label": "small_chunks_3results"},
        {"chunk_size_words": 400, "search_num_results": 5, "label": "medium_chunks_5results"},
        {"chunk_size_words": 600, "search_num_results": 7, "label": "large_chunks_7results"},
    ]

    print("\n=== TUNING RUN ===")
    tune_results = []
    for cfg_override in configs:
        label = cfg_override.pop("label")
        print(f"\n--- Config: {label} ---")
        summary = run_eval(cfg_overrides=cfg_override)
        tune_results.append({
            "label": label,
            "config": cfg_override,
            "pass_rate_pct": summary["pass_rate_pct"],
            "passed": summary["passed"],
            "total": summary["total"],
        })

    print("\n=== TUNING SUMMARY ===")
    for r in tune_results:
        print(f"  {r['label']}: {r['passed']}/{r['total']} ({r['pass_rate_pct']}%)")

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"tuning_{ts}.json"
    out.write_text(json.dumps(tune_results, indent=2))
    print(f"\nTuning results saved to: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run tuning comparison across configs")
    args = parser.parse_args()

    if args.tune:
        run_tuning()
    else:
        run_eval()
