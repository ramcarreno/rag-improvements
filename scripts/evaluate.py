import argparse
import pathlib
from collections import defaultdict
from typing import Any

import chromadb
import numpy as np
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

from utils.logger import get_logger
logger = get_logger("eval_logger", pathlib.Path("logs/evaluate.log"))

from models.baseline import BaselineRAG
from models.improved import ImprovedRAG


def rr_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    """
    Calculate reciprocal rank of first relevant documents @k retrieval.

    Args:
        retrieved_ids: List of retrieved document ids.
        gold_ids: Set of relevant document ids.
        k: Number of top relevant documents.

    Returns:
        float: Reciprocal rank metric for a single query.
    """
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in gold_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    """
    Calculate normalized discounted cumulative gain @k with binary relevance.

    Args:
        retrieved_ids: List of retrieved document ids.
        gold_ids: Set of relevant document ids.
        k: Number of top relevant documents.

    Returns:
        float: nDCG metric for a single query.
    """
    dcg: float = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in gold_ids:
            dcg += 1.0 / np.log2(i + 1)
    ideal_rels: int = min(len(gold_ids), k)
    idcg: float = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_rels + 1))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    """
    Reproduce evaluation results for the selected RAG model.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate RAG model on test data."
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[5, 10],
        help="List of k values for @k evaluation metrics: number of top "
             "retrieved documents considered."
    )
    parser.add_argument(
        "--index_path",
        type=str,
        required=True,
        help="Path to vector store index."
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="bright_documents_biology",
        help="Name of the subset to select from the vector store."
    )
    parser.add_argument(
        "--hf_subset",
        type=str,
        default="examples",
        help="Hugging Face dataset subset to load for evaluation."
    )
    parser.add_argument(
        "--hf_split",
        type=str,
        default="biology",
        help="Name of the dataset split to evaluate on."
    )
    parser.add_argument(
        "--rag_model",
        type=str,
        choices=["baseline", "improved"],
        help="RAG model to use."
    )
    parser.add_argument(
        "--embeddings_model",
        type=str,
        default="text-embedding-3-small",
        help="Model to embed queries."
    )
    parser.add_argument(
        "--embedding_batch_size",
        type=int,
        default=32,
        help="Number of queries to embed at once when calling the OpenAI "
             "embeddings API."
    )
    args = parser.parse_args()

    # Arguments correctly parsed - start logging
    logger.info(f"{'*' * 10} RAG EVALUATION {'*' * 10}")
    logger.info(f"Using model: [{args.rag_model}]")
    logger.info(f"Evaluating for k values: {args.k_values}")

    # Load ChromaDB client
    client = chromadb.PersistentClient(path=pathlib.Path(args.index_path))
    collection = client.get_collection(name=args.collection_name)

    # Load OpenAI client
    client = OpenAI()

    # Load evaluation dataset (set to BRIGHT) and subset/split
    dataset = load_dataset("xlangai/BRIGHT", args.hf_subset,
                           split=args.hf_split)

    # Instantiate RAG model
    if args.rag_model == "baseline":
        rag_model = BaselineRAG(
            retriever=collection,
            embeddings_model=args.embeddings_model,
            client=client,
        )
    elif args.rag_model == "improved":
        rag_model = ImprovedRAG(
            retriever=collection,
            embeddings_model=args.embeddings_model,
            client=client,
        )
    else:
        # This should never happen since argparse restricts the values
        logger.error(f"Unexpected rag_model value: {args.rag_model}")
        raise ValueError(f"Unexpected rag_model value: {args.rag_model}")

    # Propagate logger to RAG model
    rag_model.logger = logger

    # Look for cached embeddings, compute in batch and save not cached
    queries: list[str] = [sample["query"] for sample in dataset]
    rag_model.embed_queries(
        queries=queries,
        batch_size=args.embedding_batch_size
    )

    # Init metrics accumulators
    metrics_values = defaultdict(list)

    # Start evaluation loop
    for sample in tqdm(dataset, desc="Evaluating..."):
        # Retrieve up to top maximum k value
        result: dict[str, Any] = rag_model.retrieve(
            query_text=sample["query"],
            k=max(args.k_values)
        )
        # Extract retrievals & references for comparison
        retrieved_ids: list[str] = result["ids"][0]
        gold_ids: set[str] = set(sample["gold_ids"])

        # Calculate metrics for each k value
        for k in args.k_values:
            top_k: set[str] = set(retrieved_ids[:k])
            tp: int = len(top_k & gold_ids)

            recall: float = tp / len(gold_ids) if gold_ids else 0.0
            hit: float = 1.0 if tp > 0 else 0.0

            # Store per-sample rank-unaware metrics
            metrics_values[f"recall@{k}"].append(recall)
            metrics_values[f"hit@{k}"].append(hit)

            # Store per-sample ranked metrics
            metrics_values[f"MRR@{k}"].append(
                rr_at_k(retrieved_ids, gold_ids, k)
            )
            metrics_values[f"nDCG@{k}"].append(
                ndcg_at_k(retrieved_ids, gold_ids, k)
            )

    # Compute mean aggregates of all metrics
    final_results: dict[str, float] = {
        metric: float(np.mean(values))
        for metric, values in metrics_values.items()
    }

    # Pretty print final results, sorting metrics by k
    print(f"\nEvaluation results for [{args.rag_model}] model:")
    print(f"{'Metric':<15} | {'Value':>6}")

    last_k: int = 0
    for metric in sorted(
            final_results.keys(),
            key=lambda x: int(x.split("@")[1])
    ):
        k = int(metric.split("@")[1])
        if k != last_k:
            print("-" * 25)  # Divider between different k
        last_k = k
        print(f"{metric:<15} | {final_results[metric]:>6.3f}")


if __name__ == "__main__":
    main()