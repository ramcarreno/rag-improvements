import argparse
import pathlib

import chromadb
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

from models.baseline import BaselineRAG


def main():
    """
    Reproduce evaluation results for the selected RAG model.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate RAG model on test data."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Cutoff for evaluation metrics: number of top retrieved "
             "documents considered."
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
        raise NotImplementedError
    else:
        # This should never happen since argparse restricts the values
        raise ValueError(f"Unexpected rag_model value: {args.rag_model}")

    # Look for cached embeddings, compute in batch and save not cached
    queries: list[str] = [sample["query"] for sample in dataset]
    query_embeddings: dict[str, list[float]] = rag_model.embed_queries(
        queries=queries, batch_size=args.embedding_batch_size
    )

    # Start evaluation loop
    for sample in tqdm(dataset, desc="Evaluating"):
        result = rag_model.retrieve(query_text=sample["query"], k=args.k)
        retrieved_ids = result["ids"][0]
        gold_ids = sample["gold_ids"]
        # TODO: Metrics


if __name__ == "__main__":
    main()