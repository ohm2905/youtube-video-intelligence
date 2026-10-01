import os
import time
import json
import logging
from typing import List, Dict, Any
from pathlib import Path

from backend.app.services.rag import RAGService
from backend.app.services.vector_store import VectorStoreService
from backend.app.services.llm import LLMService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("llm_benchmark")


def load_dataset() -> List[Dict[str, Any]]:
    dataset_path = Path(__file__).parent / "dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_llm_benchmark(models: List[str] = None):
    llm = LLMService()
    if not llm.is_available():
        logger.error("Ollama server is not available. Ensure 'ollama serve' is running.")
        return

    available_models = llm.list_models()
    logger.info(f"Available local models in Ollama: {available_models}")

    target_models = models or [m for m in available_models if "llama3.2" in m or "qwen" in m or "mistral" in m]
    if not target_models and available_models:
        target_models = [available_models[0]]

    if not target_models:
        logger.warning("No target models found in Ollama to benchmark.")
        return

    dataset = load_dataset()
    vector_store = VectorStoreService.get_instance()
    rag_service = RAGService(vector_store=vector_store, llm_service=llm)

    all_results = []

    for model_name in target_models:
        logger.info(f"\n=======================================================")
        logger.info(f"Benchmarking Generation LLM: {model_name}")
        logger.info(f"=======================================================")

        correct_hallucination_catches = 0
        total_hallucination_tests = 0
        total_positive_answered = 0
        total_positive_tests = 0
        timestamp_citations_count = 0
        latencies = []

        for item in dataset:
            video_id = item["video_id"]
            question = item["question"]
            is_covered = item["is_covered"]

            t0 = time.perf_counter()
            response = rag_service.answer_question(video_id=video_id, question=question, model=model_name)
            latency = time.perf_counter() - t0
            latencies.append(latency)

            answer = response.answer
            has_timestamp_citation = (
                "[" in answer and (":" in answer or "min" in answer)
            ) or len(response.sources) > 0
            if has_timestamp_citation:
                timestamp_citations_count += 1

            if not is_covered:
                total_hallucination_tests += 1
                # The model should state that topic was not found
                if "not found in this video" in answer.lower() or "not mentioned" in answer.lower() or "not covered" in answer.lower():
                    correct_hallucination_catches += 1
                else:
                    logger.warning(f"Hallucination detected on question: '{question}'\nAnswer: {answer[:150]}")
            else:
                total_positive_tests += 1
                if "not found in this video" not in answer.lower():
                    total_positive_answered += 1

        hallucination_resistance = (
            (correct_hallucination_catches / total_hallucination_tests) * 100
            if total_hallucination_tests
            else 100
        )
        answer_rate = (
            (total_positive_answered / total_positive_tests) * 100
            if total_positive_tests
            else 0
        )
        citation_rate = (
            (timestamp_citations_count / len(dataset)) * 100 if dataset else 0
        )
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        summary = {
            "model": model_name,
            "hallucination_resistance": f"{hallucination_resistance:.1f}%",
            "grounded_answer_rate": f"{answer_rate:.1f}%",
            "timestamp_citation_rate": f"{citation_rate:.1f}%",
            "avg_latency_s": f"{avg_latency:.2f}s",
        }
        all_results.append(summary)

    # Print summary
    print("\n\n" + "=" * 80)
    print("LLM GENERATION BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Model':<30} | {'Anti-Hallucination':<20} | {'Grounded Answer':<16} | {'Citation Rate':<15} | {'Avg Latency':<12}")
    print("-" * 100)
    for r in all_results:
        print(f"{r['model']:<30} | {r['hallucination_resistance']:<20} | {r['grounded_answer_rate']:<16} | {r['timestamp_citation_rate']:<15} | {r['avg_latency_s']:<12}")
    print("=" * 80 + "\n")

    # Save Markdown report
    os.makedirs("./benchmark/results", exist_ok=True)
    report_file = "./benchmark/results/llm_benchmark_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# LLM Generation Benchmark Report\n\n")
        f.write("| Model | Anti-Hallucination Score | Grounded Answer Rate | Timestamp Citation Rate | Avg Latency |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for r in all_results:
            f.write(f"| `{r['model']}` | {r['hallucination_resistance']} | {r['grounded_answer_rate']} | {r['timestamp_citation_rate']} | {r['avg_latency_s']} |\n")
        f.write("\n*Benchmarked using strict anti-hallucination prompt on YouTube Q&A evaluation dataset.*\n")

    logger.info(f"Saved LLM benchmark report to {report_file}")


if __name__ == "__main__":
    run_llm_benchmark()
