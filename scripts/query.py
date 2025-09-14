import argparse

import chromadb
from openai import OpenAI

from models.baseline import BaselineRAG


def main():
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
        type=str,
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
        help="Model to embed queries."
    )
    args = parser.parse_args()

    # Load ChromaDB client
    client = chromadb.PersistentClient(path=args.index_path)
    collection = client.get_collection(name=args.collection_name)

    # Load OpenAI client
    client = OpenAI()

    # Instantiate RAG model
    if args.rag_model == "baseline":
        rag_model = BaselineRAG(
            retriever=collection,
            embeddings_model=args.embeddings_model,
            llm_model=args.llm_model,
            client=client,
        )
    elif args.rag_model == "improved":
        raise NotImplementedError
    else:
        # This should never happen since argparse restricts the values
        raise ValueError(f"Unexpected rag_model value: {args.rag_model}")

    # Run query and print the answer
    rag_model.answer(query=args.query_text, k=args.k)


if __name__ == '__main__':
    main()
