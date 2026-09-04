import argparse
import csv
import json
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from dotenv import find_dotenv, load_dotenv
from groq import RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

load_dotenv(find_dotenv())

# ragas's new metrics API needs an OpenAI-style structured-output client, no
# Groq/HF support yet -- sticking with the classic (deprecated) metrics API
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, message=r"Importing .* from 'ragas\.metrics' is deprecated"
)

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from ragas.run_config import RunConfig

# Groq models are less reliable than GPT at ragas's strict JSON format, which
# otherwise scores a question as NaN on the first bad parse. Kept modest since
# retries cost extra tokens and context_precision/recall are already expensive.
METRIC_MAX_RETRIES = 2

from src.config import DB_FAISS_PATH, VECTORSTORE_METADATA_PATH, AppConfig, get_cli_config
from src.rag import build_chain, get_embedding_model, get_llm

DEFAULT_DATASET_PATH = Path(__file__).parent / "eval" / "golden_qa.json"
DEFAULT_RESULTS_DIR = Path(__file__).parent / "eval" / "results"
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
MLFLOW_EXPERIMENT_NAME = "healthcare-chatbot-rag-eval"

# context_precision/context_recall call the judge once per retrieved chunk
# (not once per question) so they're much pricier -- skippable via --metrics
METRIC_FACTORIES = {
    "faithfulness": lambda: Faithfulness(max_retries=METRIC_MAX_RETRIES),
    "answer_relevancy": lambda: AnswerRelevancy(strictness=1),
    "context_precision": lambda: ContextPrecision(max_retries=METRIC_MAX_RETRIES),
    "context_recall": lambda: ContextRecall(max_retries=METRIC_MAX_RETRIES),
}


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def append_history(
    output_dir: Path,
    config: AppConfig,
    summary: dict[str, float],
    dataset_path: Path,
    question_count: int = 0,
    metrics_run: str = "",
) -> Path:
    history_path = output_dir / "history.csv"
    fieldnames = [
        "timestamp_utc",
        "git_commit",
        "dataset",
        "questions",  # without this a 6-question smoke run looks the same as a full run
        "metrics",
        "model_name",
        "retrieval_k",
        "rerank_k",
        "enable_reranking",
        *METRIC_NAMES,
    ]
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": get_git_commit(),
        "dataset": dataset_path.name,
        "questions": question_count,
        "metrics": metrics_run,
        "model_name": config.model_name,
        "retrieval_k": config.retrieval_k,
        "rerank_k": config.rerank_k,
        "enable_reranking": config.enable_reranking,
        **{metric: round(summary.get(metric, float("nan")), 4) for metric in METRIC_NAMES},
    }

    # migrate old rows if a column got added, otherwise new rows misalign under the old header
    existing_rows: list[dict] = []
    needs_migration = False
    if history_path.exists():
        with history_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and list(reader.fieldnames) != fieldnames:
                existing_rows = list(reader)
                needs_migration = True

    if needs_migration:
        with history_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for old_row in existing_rows:
                writer.writerow({key: old_row.get(key, "") for key in fieldnames})
            writer.writerow(row)
        return history_path

    write_header = not history_path.exists()
    with history_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return history_path


def load_golden_dataset(path: Path) -> list[dict]:
    items = json.loads(path.read_text(encoding="utf-8"))
    for item in items:
        if "question" not in item or "ground_truth" not in item:
            raise ValueError(f"Golden dataset entry missing 'question' or 'ground_truth': {item}")
    return items


# Groq's free tier caps tokens-per-minute (not just per-day), so a burst of
# back-to-back RAG calls can 429 well before the daily budget is used up.
# Retrying after a fixed wait is simpler and safer here than parsing the
# server's suggested backoff out of the error message.
@retry(
    retry=retry_if_exception_type(RateLimitError),
    stop=stop_after_attempt(5),
    wait=wait_fixed(65),
    reraise=True,
)
def _invoke_with_backoff(chain, question: str):
    return chain.invoke({"question": question, "chat_history": []})


def run_pipeline(config: AppConfig, golden: list[dict]) -> list[SingleTurnSample]:
    chain = build_chain(config)
    samples = []
    for item in golden:
        response = _invoke_with_backoff(chain, item["question"])
        answer = (response.get("answer") or "").strip()
        contexts = [doc.page_content for doc in response.get("source_documents", [])]
        samples.append(
            SingleTurnSample(
                user_input=item["question"],
                response=answer or "No answer produced.",
                retrieved_contexts=contexts or ["No context retrieved."],
                reference=item["ground_truth"],
            )
        )
        print(f"  answered: {item['id']}")
        time.sleep(3)  # stay clear of the per-minute token ceiling
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Run a RAGAS evaluation against the medical chatbot's live RAG pipeline."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="Path to the golden Q&A JSON file.")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Directory to write results.csv and summary.json."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N questions from the dataset. Useful for cheap iteration against Groq's free-tier token limits.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=",".join(METRIC_NAMES),
        help=(
            "Comma-separated subset of metrics to run: "
            f"{', '.join(METRIC_NAMES)}. context_precision/context_recall call the judge "
            "once per retrieved chunk and are the most token-expensive -- drop them for a "
            "cheap smoke check."
        ),
    )
    args = parser.parse_args()

    selected_metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    unknown_metrics = [m for m in selected_metrics if m not in METRIC_FACTORIES]
    if unknown_metrics:
        sys.exit(f"Unknown metric(s): {', '.join(unknown_metrics)}. Valid options: {', '.join(METRIC_NAMES)}")

    if not DB_FAISS_PATH.exists():
        sys.exit(f"Vector store not found at {DB_FAISS_PATH}. Run create_memory_for_llm.py first.")

    config = get_cli_config()
    golden = load_golden_dataset(args.dataset)
    if args.limit is not None:
        golden = golden[: args.limit]

    vectorstore_metadata = {}
    if VECTORSTORE_METADATA_PATH.exists():
        vectorstore_metadata = json.loads(VECTORSTORE_METADATA_PATH.read_text(encoding="utf-8"))

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"{get_git_commit()}-{args.dataset.stem}"):
        mlflow.log_params(
            {
                "git_commit": get_git_commit(),
                "dataset": args.dataset.name,
                "dataset_size": len(golden),
                "model_name": config.model_name,
                "temperature": config.temperature,
                "retrieval_k": config.retrieval_k,
                "rerank_k": config.rerank_k,
                "enable_reranking": config.enable_reranking,
                "embedding_model": vectorstore_metadata.get("embedding_model"),
                "chunk_size": vectorstore_metadata.get("chunk_size"),
                "chunk_overlap": vectorstore_metadata.get("chunk_overlap"),
                "metrics": ",".join(selected_metrics),
            }
        )

        print(f"Running {len(golden)} golden questions through the live RAG pipeline...")
        samples = run_pipeline(config, golden)

        # faithfulness needs a long structured response from the judge; without
        # max_tokens Groq's default cap was truncating it and tanking the score
        judge_llm = get_llm(config.model_name, config.groq_api_key, 0.0, max_tokens=4096)

        print(f"Scoring with RAGAS ({', '.join(selected_metrics)})...")
        # strictness=1 on AnswerRelevancy: the default (3) makes ragas request
        # n=3 completions in a single call; Groq's API rejects any n > 1 with
        # a 400, which silently became a NaN for this metric on almost every
        # question.
        metrics = [METRIC_FACTORIES[name]() for name in selected_metrics]
        result = evaluate(
            dataset=EvaluationDataset(samples=samples),
            metrics=metrics,
            llm=LangchainLLMWrapper(judge_llm),
            embeddings=LangchainEmbeddingsWrapper(get_embedding_model()),
            # Serialize + back off generously: ragas's default max_workers=16 would
            # fire concurrent judge calls and blow through Groq's free-tier 12K
            # tokens-per-minute cap almost immediately.
            run_config=RunConfig(max_workers=1, max_retries=10, max_wait=90),
        )

        df = result.to_pandas()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_dir / "results.csv", index=False)

        summary = {metric: float(df[metric].mean(skipna=True)) for metric in METRIC_NAMES if metric in df.columns}
        sample_counts = {metric: int(df[metric].notna().sum()) for metric in METRIC_NAMES if metric in df.columns}
        (args.output_dir / "summary.json").write_text(
            json.dumps({"mean": summary, "scored_of_total": {k: f"{v}/{len(df)}" for k, v in sample_counts.items()}}, indent=2),
            encoding="utf-8",
        )
        history_path = append_history(
            args.output_dir, config, summary, args.dataset,
            question_count=len(golden), metrics_run=",".join(selected_metrics),
        )

        mlflow.log_metrics({metric: value for metric, value in summary.items() if value == value})  # skip NaN
        mlflow.log_metrics({f"{metric}_sample_count": count for metric, count in sample_counts.items()})
        mlflow.log_artifact(str(args.output_dir / "results.csv"))
        mlflow.log_artifact(str(args.output_dir / "summary.json"))

        print("\n=== Per-question scores ===")
        score_columns = [m for m in METRIC_NAMES if m in df.columns]
        print(df[["user_input", *score_columns]].to_string(index=False))

        print("\n=== Mean scores ===")
        for metric, value in summary.items():
            n = sample_counts[metric]
            flag = "  <-- based on a small sample, treat with caution" if n < len(df) * 0.7 else ""
            print(f"{metric:20s}: {value:.3f}  (scored {n}/{len(df)} questions){flag}")

        print(f"\nSaved detailed results to {args.output_dir / 'results.csv'}")
        print(f"Saved summary to {args.output_dir / 'summary.json'}")
        print(f"Appended run to {history_path}")
        print(f"Logged run to MLflow experiment '{MLFLOW_EXPERIMENT_NAME}' (run mlflow ui to inspect)")


if __name__ == "__main__":
    main()
