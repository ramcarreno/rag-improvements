import argparse
import pathlib
import sys
import time

import chromadb
from openai import OpenAI

from utils.logger import get_logger
logger = get_logger("query_logger", pathlib.Path("logs/query.log"))

from models.baseline import BaselineRAG
from models.improved import ImprovedRAG


def build_parser() -> argparse.ArgumentParser:
    """
    Argument parser for CLI.

    Returns:
        argparse.ArgumentParser: The parser instance with the corresponding
            arguments.
    """
    # Declare and parse all possible CLI arguments
    parser = argparse.ArgumentParser(
        description="Run a retrieval query and get an answer from an LLM."
    )
    parser.add_argument(
        "--query_text",
        type=str,
        required=True,
        help="The query to run."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of documents to retrieve."
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
        "--llm_model",
        type=str,
        default="gpt-5-mini",
        help="Model to generate text responses."
    )
    return parser


def main(argv: list[str] | None = None) -> str:
    """
    Run a user-given query for the selected RAG model.

    Returns:
        str: The LLM output of the query.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Arguments correctly parsed - start logging
    logger.info(f"{'*' * 10} RAG QUERY {'*' * 10}")
    logger.info(f"Using model: [{args.rag_model}]")
    logger.info(f"Query: [{args.query_text}]")
    logger.info(f"Retrieving top k={args.k} documents")

    # Load ChromaDB client
    try:
        client = chromadb.PersistentClient(path=pathlib.Path(args.index_path))
        collection = client.get_collection(name=args.collection_name)
    except Exception as e:
        logger.exception("Something went wrong when loading the ChromaDB "
                         "index.")
        raise

    # Load OpenAI client
    try:
        client = OpenAI()
    except Exception as e:
        logger.exception("Something went wrong when loading the OpenAI "
                         "client.")
        raise

    # Instantiate RAG model
    if args.rag_model == "baseline":
        rag_model = BaselineRAG(
            retriever=collection,
            embeddings_model=args.embeddings_model,
            llm_model=args.llm_model,
            client=client,
        )
    elif args.rag_model == "improved":
        rag_model = ImprovedRAG(
            retriever=collection,
            embeddings_model=args.embeddings_model,
            llm_model=args.llm_model,
            client=client,
        )
    else:
        # This should never happen since argparse restricts the values
        logger.error(f"Unexpected rag_model value: {args.rag_model}")
        raise ValueError(f"Unexpected rag_model value: {args.rag_model}")

    # Propagate logger to RAG model
    rag_model.logger = logger

    # Run query and print the answer
    return rag_model.answer(query_text=args.query_text, k=args.k)


if __name__ == "__main__":
    # Set up timer
    start_time = time.perf_counter()
    try:
        answer = main()
        end_time = time.perf_counter()
        elapsed = end_time - start_time

        print(answer)
        logger.info(f"Answer: [{answer}]")
        logger.info(f"Answer generated in {elapsed:.2f} seconds")
    except Exception as e:
        logger.error("Query failed!")
        sys.exit(1)