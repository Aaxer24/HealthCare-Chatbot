import argparse
import csv
import json
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from groq import RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

load_dotenv(find_dotenv())

# ragas.metrics still supports these classic metric instances via a deprecation
# shim (ragas.metrics.collections is the new API, but it requires an
# instructor/OpenAI-style structured-output client and has no Groq/HF support
# yet). The classic API works fine with our existing ChatGroq + HuggingFace
# embeddings stack, so we deliberately keep using it.
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, message=r"Importing .* from 'ragas\.metrics' is deprecated"
)

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from ragas.run_config import RunConfig

# Groq's Llama models are less consistent than GPT at following ragas's strict
# JSON output format for these three metrics, which otherwise silently scores
# a question as NaN after a single failed parse. max_retries makes ragas send
# the malformed output back to the LLM with a "fix this to match the schema"
# prompt before giving up, which is the standard mitigation for this gap.
# Kept modest (not higher) because each retry is extra tokens against Groq's
# free-tier daily cap, and context_precision/recall already call the LLM once
# per retrieved chunk.
METRIC_MAX_RETRIES = 2

from src.config import DB_FAISS_PATH, AppConfig, get_cli_config
from src.rag import build_chain, get_embedding_model, get_llm

DEFAULT_DATASET_PATH = Path(__file__).parent / "eval" / "golden_qa.json"
DEFAULT_RESULTS_DIR = Path(__file__).parent / "eval" / "results"
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


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


def append_history(output_dir: Path, config: AppConfig, summary: dict[str, float], dataset_path: Path) -> Path:
    history_path = output_dir / "history.csv"
    fieldnames = [
        "timestamp_utc",
        "git_commit",
        "dataset",
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
        "model_name": config.model_name,
        "retrieval_k": config.retrieval_k,
        "rerank_k": config.rerank_k,
        "enable_reranking": config.enable_reranking,
        **{metric: round(summary.get(metric, float("nan")), 4) for metric in METRIC_NAMES},
    }

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
    args = parser.parse_args()

    if not DB_FAISS_PATH.exists():
        sys.exit(f"Vector store not found at {DB_FAISS_PATH}. Run create_memory_for_llm.py first.")

    config = get_cli_config()
    golden = load_golden_dataset(args.dataset)
    if args.limit is not None:
        golden = golden[: args.limit]

    print(f"Running {len(golden)} golden questions through the live RAG pipeline...")
    samples = run_pipeline(config, golden)

    judge_llm = get_llm(config.model_name, config.groq_api_key, 0.0)

    print("Scoring with RAGAS (faithfulness, answer relevancy, context precision, context recall)...")
    metrics = [
        Faithfulness(max_retries=METRIC_MAX_RETRIES),
        # strictness (default 3) makes ragas request n=3 completions in a
        # single call; Groq's API rejects any n > 1 with a 400, which silently
        # became a NaN for this metric on almost every question. strictness=1
        # avoids the unsupported parameter entirely.
        AnswerRelevancy(strictness=1),
        ContextPrecision(max_retries=METRIC_MAX_RETRIES),
        ContextRecall(max_retries=METRIC_MAX_RETRIES),
    ]
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
    history_path = append_history(args.output_dir, config, summary, args.dataset)

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


if __name__ == "__main__":
    main()
