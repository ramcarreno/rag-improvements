import json
import pathlib
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from openai import OpenAI
from tqdm import tqdm


class RAGModel(ABC):
    """
    Abstract base class for all RAG models.
    """
    def __init__(self, retriever: Any,
                 embeddings_model: str = "text-embedding-3-small",
                 llm_model: str | None = None, client: OpenAI | None = None):
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

        self.cache_file = pathlib.Path(".emb_cache/query_embeddings.json")
        self.cache_file.parent.mkdir(exist_ok=True)
        self.embeddings_cache = self._load_embeddings_cache()

    def _load_embeddings_cache(self) -> dict[str, list[float]]:
        """
        Load embeddings cache file.

        Returns:
            dict[str, list[float]]: Dict loading embeddings from cache file.
        """
        if self.cache_file.exists():
            return json.loads(self.cache_file.read_text())
        return {}

    def embed_query(self, query_text: str) -> list[float]:
        """
        Generate OpenAI embeddings for a RAG query.
        Reads embeddings cache before generating.

        Args:
            query_text (str): RAG query in text format.

        Returns:
            list[float]: Text embedding vector.
        """
        return self.embed_queries([query_text])[query_text]

    def embed_queries(self, queries: list[str], batch_size: int = 32) \
            -> dict[str, list[float]]:
        """
        Embed a list of queries using cache if available, otherwise call
        OpenAI API in batches to generate embeddings.

        Args:
            queries (list[str]): A list of text queries.
            batch_size (int): Number of queries per batch.

        Returns:
            dict[str, list[float]]: Query-embeddings key-value pairs.
        """
        # Filter queries not found in embeddings cache
        to_embed = [q for q in queries if q not in self.embeddings_cache]

        # Batch embedding for new queries
        if to_embed:
            for i in tqdm(range(0, len(to_embed), batch_size),
                          desc="[Generating embeddings...]"):
                batch = to_embed[i:i+batch_size]
                response = self.client.embeddings.create(
                    input=batch, model=self.embeddings_model
                )
                # Store each new entry in cache dict
                for q, emb_data in zip(batch, response.data):
                    self.embeddings_cache[q] = np.array(
                        emb_data.embedding,
                        dtype=np.float32
                    ).tolist()

        # Save updated cache from dict
        self.cache_file.write_text(json.dumps(self.embeddings_cache))

        # Return only queried embeddings
        return {q: self.embeddings_cache[q] for q in queries}

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

