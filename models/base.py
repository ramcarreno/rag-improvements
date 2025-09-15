from abc import ABC, abstractmethod
from typing import Any

from openai import OpenAI


class RAGModel(ABC):
    """
    Abstract base class for all RAG models.
    """
    def __init__(self, retriever: Any,
                 embeddings_model: str = "text-embedding-3-small",
                 llm_model: str = "gpt-5-mini", client: OpenAI | None = None):
        """
        Args:
            retriever (Any): Retriever for RAG models - returns documents when
                queried.
            embeddings_model (str): Name of model that will embed the queries.
            llm_model (str): Name of the large language model that will
                produce answers.
            client (OpenAI | None): OpenAI client for embeddings / LLM calls.
        """
        self.retriever = retriever
        self.embeddings_model = embeddings_model
        self.llm_model = llm_model
        self.client = client or OpenAI()

    @abstractmethod
    def retrieve(self, query_text: str, k: int) -> dict[str, Any]:
        """
        Retrieve a list of k documents from a vector database based on a
        query.

        Args:
            query_text (str): Query in textual format to be converted into an
                embedding vector and sent to the retriever.
            k (int): Number of documents to retrieve.

        Returns:
            dict[str, Any]: A dictionary in the format returned by ChromaDB's
                `collection.query()`, including lists of retrieved documents,
                their ids, distances to query and metadata.
        """

    @abstractmethod
    def answer(self, query_text: str, k: int) -> str:
        """
        Answer a query with an LLM based on previously k retrieved documents.

        Args:
            query_text (str): Query to be sent to the LLM.
            k (int): Number of documents to be used in retrieval to generate
                answers.

        Returns:
            str: LLM RAG-informed answer.
        """
