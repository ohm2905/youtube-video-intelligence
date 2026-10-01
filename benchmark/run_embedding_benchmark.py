import os
import time
import json
import logging
from typing import List, Dict, Any
from pathlib import Path

from backend.app.services.youtube import fetch_transcript, fetch_video_metadata
from backend.app.services.chunker import create_timestamped_chunks
from backend.app.services.embedding import EmbeddingService
from backend.app.services.vector_store import VectorStoreService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")

BENCHMARK_MODELS = [
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
]


def load_dataset() -> List[Dict[str, Any]]:
    dataset_path = Path(__file__).parent / "dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_timestamp_overlap(chunk_start: float, chunk_end: float, expected_start: float, expected_end: float) -> bool:
    """Check if the retrieved chunk time interval overlaps with the ground truth expected interval."""
    return max(chunk_start, expected_start) <= min(chunk_end, expected_end)


def run_benchmark():
    dataset = load_dataset()
    logger.info(f"Loaded benchmark dataset with {len(dataset)} items.")

    # Unique videos in dataset
    video_ids = list(set(item["video_id"] for item in dataset))
    
    # Pre-fetch and chunk transcripts
    video_chunks_map = {}
    for vid in video_ids:
        logger.info(f"Fetching transcript for video {vid}...")
        snippets = fetch_transcript(vid)
        chunks = create_timestamped_chunks(snippets, chunk_size=250, chunk_overlap=40)
        video_chunks_map[vid] = chunks
        logger.info(f"Video {vid}: created {len(chunks)} chunks.")

    results_summary = []

    for model_name in BENCHMARK_MODELS:
        logger.info(f"\n=======================================================")
        logger.info(f"Benchmarking Embedding Model: {model_name}")
        logger.info(f"=======================================================")

        # Initialize dedicated vector store for this model benchmark
        benchmark_store_dir = f"./data/benchmark_chroma_{model_name.replace('/', '_')}"
        vector_store = VectorStoreService(persist_directory=benchmark_store_dir)
        embedder = EmbeddingService(model_name=model_name)

        # Index videos
        for vid, chunks in video_chunks_map.items():
            vector_store.delete_video_chunks(vid)
            vector_store.index_video_chunks(vid, chunks, embedding_service=embedder)

        # Run queries
        k_values = [1, 3, 5]
        hits_at_k = {k: 0 for k in k_values}
        mrr_total = 0.0
        positive_queries = [q for q in dataset if q["is_covered"]]
        num_pos = len(positive_queries)

        latencies = []

        for item in positive_queries:
            t0 = time.perf_counter()
            retrieved = vector_store.search_video_chunks(
                video_id=item["video_id"],
                query=item["question"],
                top_k=max(k_values),
                embedding_service=embedder,
            )
            query_time_ms = (time.perf_counter() - t0) * 1000
            latencies.append(query_time_ms)

            # Evaluate recall and MRR
            found_rank = None
            for rank, chunk in enumerate(retrieved, start=1):
                if check_timestamp_overlap(
                    chunk.start_time,
                    chunk.end_time,
                    item["expected_start_time"],
                    item["expected_end_time"],
                ):
                    found_rank = rank
                    break

            if found_rank is not None:
                for k in k_values:
                    if found_rank <= k:
                        hits_at_k[k] += 1
                mrr_total += 1.0 / found_rank

        recall_at_1 = (hits_at_k[1] / num_pos) * 100 if num_pos else 0
        recall_at_3 = (hits_at_k[3] / num_pos) * 100 if num_pos else 0
        recall_at_5 = (hits_at_k[5] / num_pos) * 100 if num_pos else 0
        mrr = (mrr_total / num_pos) if num_pos else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        summary = {
            "model": model_name,
            "dimension": embedder.dimension,
            "recall@1": f"{recall_at_1:.1f}%",
            "recall@3": f"{recall_at_3:.1f}%",
            "recall@5": f"{recall_at_5:.1f}%",
            "mrr": f"{mrr:.3f}",
            "avg_latency_ms": f"{avg_latency:.2f} ms",
        }
        results_summary.append(summary)

    # Print summary table
    print("\n\n" + "=" * 80)
    print("EMBEDDING MODEL BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Model':<40} | {'Dim':<5} | {'Recall@1':<9} | {'Recall@3':<9} | {'Recall@5':<9} | {'MRR':<6} | {'Avg Latency':<12}")
    print("-" * 100)
    for r in results_summary:
        print(f"{r['model']:<40} | {r['dimension']:<5} | {r['recall@1']:<9} | {r['recall@3']:<9} | {r['recall@5']:<9} | {r['mrr']:<6} | {r['avg_latency_ms']:<12}")
    print("=" * 80 + "\n")

    # Save to Markdown report
    os.makedirs("./benchmark/results", exist_ok=True)
    report_file = "./benchmark/results/embedding_benchmark_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Embedding Model Benchmark Report\n\n")
        f.write("| Model | Dimension | Recall@1 | Recall@3 | Recall@5 | MRR | Avg Query Latency |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in results_summary:
            f.write(f"| `{r['model']}` | {r['dimension']} | {r['recall@1']} | {r['recall@3']} | {r['recall@5']} | {r['mrr']} | {r['avg_latency_ms']} |\n")
        f.write("\n*Benchmarked against ground truth timestamped YouTube video intervals.*\n")

    logger.info(f"Saved benchmark report to {report_file}")


if __name__ == "__main__":
    run_benchmark()
