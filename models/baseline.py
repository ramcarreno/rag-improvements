from typing import Any

from openai import OpenAI

from models.base import RAGModel


class BaselineRAG(RAGModel):
    """
    Baseline RAG model for retrieval.
    """
    def __init__(self, retriever,
                 embeddings_model: str = "text-embedding-3-small",
                 llm_model: str | None = None, client: OpenAI | None = None):
        # Pass args up to RAGModel (base) class
        super().__init__(retriever, embeddings_model, llm_model, client)

    def retrieve(self, query_text: str, k: int) -> dict[str, Any]:
        """
        Implements simple retrieval querying ChromaDB.
        """
        # Embed query
        query_embeddings: list[float] = super().embed_query(query_text)

        # Look up on vector store
        result: dict[str, Any] = self.retriever.query(
            query_embeddings=[query_embeddings], n_results=k
        )
        return result

    def answer(self, query_text: str, k: int) -> str:
        """
        Answers the query at LLM level through a simple structured prompt.
        """
        # Return k most relevant documents
        result: dict[str, Any] = self.retrieve(query_text, k)
        documents: list[str] = result.get("documents", [[]])[0]

        # Build context and prompt from retrieval result
        rag_context: str = "\n\n".join(documents)
        user_prompt: str = (
            f"Context:\n{rag_context}\n\n"
            f"Question: {query_text}\n\n"
            "Answer clearly and concisely based only on the context provided."
        )

        # Call OpenAI API and return text response
        response = self.client.responses.create(
            model=self.llm_model,
            input=[
                {"role": "system", "content": "You are a helpful assistant "
                                              "answering biology questions."},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.output_text


