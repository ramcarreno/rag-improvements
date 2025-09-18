import logging
import pathlib
import shelve
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
                 llm_model: str | None = None, client: OpenAI | None = None,
                 logger: logging.Logger = None):
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

        self.logger = logger
        self.cache_file = ".emb_cache/query_embeddings.db"
        pathlib.Path(".emb_cache").mkdir(exist_ok=True)


    def embed_queries(self, queries: list[str], batch_size: int = 32) \
            -> dict[str, list[float]]:
        """
        Embed multiple queries using cached embeddings in a shelve DB.
        Only new queries call the embeddings API (in batches).

        Args:
            queries (list[str]): A list of text queries.
            batch_size (int): Number of queries per batch.

        Returns:
            dict[str, list[float]]: Query-embeddings key-value pairs.
        """
        embeddings: dict[str, list[float]] = {}

        with shelve.open(self.cache_file) as db:
            # Filter queries not in embeddings cache (new)
            to_embed: list[str] = [q for q in queries if q not in db]

            # Batch embed new queries
            if to_embed:
                for i in tqdm(range(0, len(to_embed), batch_size),
                              desc="[Generating embeddings...]"):
                    batch: list[str] = to_embed[i:i + batch_size]
                    try:
                        response = self.client.embeddings.create(
                            input=batch, model=self.embeddings_model
                        )
                    except Exception as e:
                        self.logger.exception(f"Failed to create embeddings "
                                              f"for batch of size "
                                              f"{len(batch)}!")
                        raise
                    # Store new entries & format for direct usage
                    for q, emb_data in zip(batch, response.data):
                        emb_vec = np.array(
                            emb_data.embedding, dtype=np.float32
                        )
                        db[q] = emb_vec
                        embeddings[q] = emb_vec.tolist()

            # Load all requested embeddings from cache
            for q in queries:
                if q not in embeddings:
                    embeddings[q] = db[q].tolist()

        return embeddings

    def embed_query(self, query_text: str) -> list[float]:
        """
        Generate embeddings for a RAG query.
        Reads embeddings cache before generating.

        Args:
            query_text (str): RAG query in text format.

        Returns:
            list[float]: Text embedding vector.
        """
        return self.embed_queries([query_text])[query_text]

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

