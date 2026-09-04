"""Review collected feedback and promote questions into the golden eval set.

This is the human checkpoint in the improvement loop. Thumbs-down answers are
the questions the pipeline handled badly, which makes them the most valuable
additions to eval/golden_qa.json -- but they are NOT added automatically:

* A ground truth must be written by hand from the source PDFs. Auto-generating
  one from the model's own (bad) answer would bake the failure into the metric.
* Not every complaint is a retrieval failure; some are questions the corpus
  genuinely cannot answer, and adding those would permanently depress scores
  for something the system was never given.

So this script only *stages* candidates. A human fills in ground_truth, then
runs --promote to merge the completed ones in.

Usage:
    python scripts/review_feedback.py --list
    python scripts/review_feedback.py --stage        # write candidates file
    python scripts/review_feedback.py --promote      # merge completed ones
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feedback import feedback_summary, get_feedback_path, load_feedback  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
GOLDEN_PATH = BASE_DIR / "eval" / "golden_qa.json"
CANDIDATES_PATH = BASE_DIR / "eval" / "feedback" / "golden_candidates.json"

GROUND_TRUTH_PLACEHOLDER = "TODO: write the correct answer using the source PDFs"


def slugify(text: str, max_words: int = 5) -> str:
    words = re.sub(r"[^a-z0-9\s-]", "", text.lower()).split()
    return "-".join(words[:max_words]) or "question"


def load_golden() -> list[dict]:
    if not GOLDEN_PATH.exists():
        return []
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def cmd_list() -> None:
    records = load_feedback()
    if not records:
        print(f"No feedback recorded yet at {get_feedback_path()}")
        return

    summary = feedback_summary()
    print(f"Total: {summary['total']}  up: {summary['up']}  down: {summary['down']}  "
          f"satisfaction: {summary['satisfaction_rate']:.1%}\n")

    negative = [r for r in records if r.get("rating") == "down"]
    if not negative:
        print("No negative feedback -- nothing to review.")
        return

    print(f"=== {len(negative)} negative ratings ===")
    for record in negative:
        print(f"\n  [{record.get('timestamp_utc', '?')}]  {record.get('question', '')}")
        answer = (record.get("answer") or "").replace("\n", " ")
        print(f"    answer: {answer[:150]}...")
        if record.get("comment"):
            print(f"    comment: {record['comment']}")


def cmd_stage() -> None:
    """Write negative-feedback questions into a candidates file for editing."""
    records = load_feedback()
    negative = [r for r in records if r.get("rating") == "down"]
    if not negative:
        print("No negative feedback to stage.")
        return

    golden = load_golden()
    existing_questions = {item["question"].strip().lower() for item in golden}
    existing_ids = {item["id"] for item in golden}

    candidates = []
    if CANDIDATES_PATH.exists():
        candidates = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))
    staged_questions = {c["question"].strip().lower() for c in candidates}

    added = 0
    for record in negative:
        question = (record.get("question") or "").strip()
        if not question:
            continue
        key = question.lower()
        # Skip anything already in the golden set or already staged, so running
        # this repeatedly is safe.
        if key in existing_questions or key in staged_questions:
            continue

        candidate_id = slugify(question)
        suffix = 2
        while candidate_id in existing_ids:
            candidate_id = f"{slugify(question)}-{suffix}"
            suffix += 1
        existing_ids.add(candidate_id)

        candidates.append({
            "id": candidate_id,
            "question": question,
            "ground_truth": GROUND_TRUTH_PLACEHOLDER,
            "_rejected_answer": record.get("answer", "")[:500],
            "_sources_at_the_time": [s.get("source", "") for s in record.get("sources", [])],
        })
        staged_questions.add(key)
        added += 1

    CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATES_PATH.write_text(json.dumps(candidates, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Staged {added} new candidate(s) -> {CANDIDATES_PATH}")
    print(f"{len(candidates)} total awaiting review.")
    print("\nNext: open that file, replace each ground_truth placeholder using the")
    print("source PDFs, then run:  python scripts/review_feedback.py --promote")


def cmd_promote() -> None:
    """Merge candidates that have a real ground truth into the golden set."""
    if not CANDIDATES_PATH.exists():
        print("No candidates file. Run --stage first.")
        return

    candidates = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))
    golden = load_golden()
    existing_questions = {item["question"].strip().lower() for item in golden}

    ready, pending = [], []
    for candidate in candidates:
        ground_truth = (candidate.get("ground_truth") or "").strip()
        if not ground_truth or ground_truth == GROUND_TRUTH_PLACEHOLDER:
            pending.append(candidate)
        elif candidate["question"].strip().lower() in existing_questions:
            print(f"  skipping duplicate: {candidate['question'][:60]}")
        else:
            # Drop the underscore-prefixed review-only fields before merging.
            ready.append({
                "id": candidate["id"],
                "question": candidate["question"],
                "ground_truth": ground_truth,
            })

    if not ready:
        print(f"Nothing ready to promote ({len(pending)} still need a ground_truth).")
        return

    golden.extend(ready)
    GOLDEN_PATH.write_text(json.dumps(golden, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    CANDIDATES_PATH.write_text(json.dumps(pending, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Promoted {len(ready)} question(s) into {GOLDEN_PATH.name} (now {len(golden)} total).")
    print(f"{len(pending)} candidate(s) still awaiting a ground_truth.")
    print("\nRe-run the evaluation to get a new baseline:")
    print("  python evaluate_chatbot.py")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="Show collected feedback")
    group.add_argument("--stage", action="store_true", help="Stage negative feedback as golden-set candidates")
    group.add_argument("--promote", action="store_true", help="Merge completed candidates into golden_qa.json")
    args = parser.parse_args()

    if args.list:
        cmd_list()
    elif args.stage:
        cmd_stage()
    else:
        cmd_promote()


if __name__ == "__main__":
    main()
