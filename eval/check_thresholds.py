"""Fail (exit 1) if any RAGAS score in summary.json drops below its floor.

CI regression gate, not a quality target -- catches a badly broken pipeline,
doesn't replace actually running evaluate_chatbot.py to improve scores.
"""
import argparse
import json
import sys
from pathlib import Path

DEFAULT_THRESHOLDS = {
    "faithfulness": 0.6,
    "answer_relevancy": 0.6,
    "context_precision": 0.5,
    "context_recall": 0.5,
}


def main():
    parser = argparse.ArgumentParser(description="Check RAGAS summary.json scores against minimum thresholds.")
    parser.add_argument("summary", type=Path, help="Path to summary.json produced by evaluate_chatbot.py")
    parser.add_argument(
        "--metrics",
        type=str,
        default=",".join(DEFAULT_THRESHOLDS),
        help="Comma-separated subset of metrics to gate on -- must match what evaluate_chatbot.py --metrics ran.",
    )
    for metric, default in DEFAULT_THRESHOLDS.items():
        parser.add_argument(f"--min-{metric.replace('_', '-')}", type=float, default=default)
    args = parser.parse_args()

    metrics_to_check = [m.strip() for m in args.metrics.split(",") if m.strip()]
    unknown = [m for m in metrics_to_check if m not in DEFAULT_THRESHOLDS]
    if unknown:
        sys.exit(f"Unknown metric(s): {', '.join(unknown)}. Valid options: {', '.join(DEFAULT_THRESHOLDS)}")

    if not args.summary.exists():
        sys.exit(f"Summary file not found: {args.summary}")

    data = json.loads(args.summary.read_text(encoding="utf-8"))
    mean = data.get("mean", {})

    failures = []
    print("=== Eval-gate thresholds ===")
    for metric in metrics_to_check:
        threshold = getattr(args, f"min_{metric}")
        score = mean.get(metric)
        if score is None:
            failures.append(f"{metric}: missing from summary.json")
            print(f"{metric:20s}: MISSING")
            continue
        status = "OK" if score >= threshold else "FAIL"
        print(f"{metric:20s}: {score:.3f}  (min {threshold:.2f})  {status}")
        if score < threshold:
            failures.append(f"{metric}: {score:.3f} < {threshold:.2f}")

    if failures:
        print("\nEval gate failed:")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)

    print("\nEval gate passed.")


if __name__ == "__main__":
    main()
