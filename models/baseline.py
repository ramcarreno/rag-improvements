from typing import Any

from openai import OpenAI

from models.base import RAGModel
from models.utils import embed_query


class BaselineRAG(RAGModel):
    def __init__(self, retriever,
                 embeddings_model: str = "text-embedding-3-small",
                 llm_model: str = "gpt-5-mini", client: OpenAI | None = None):
        # Pass args up to RAGModel (base) class
        super().__init__(retriever, embeddings_model, llm_model, client)

    def retrieve(self, query_text: str, k: int) -> dict[str, Any]:
        # Embed query
        query_embeddings = embed_query(query_text, self.embeddings_model,
                                       self.client)
        # Look up on vector store
        result = self.retriever.query(query_embeddings=[query_embeddings],
                                      n_results=k)
        return result

    def answer(self, query_text: str, k: int) -> str:
        # Return k most relevant documents
        result = self.retrieve(query_text, k)
        documents = result.get("documents", [[]])[0]

        # Build context and prompt from retrieval result
        rag_context = "\n\n".join(documents)
        user_prompt = (
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


