from openai import OpenAI

from models.base import RAGModel
from models.utils import embed_query


class BaselineRAG(RAGModel):
    def __init__(self, retriever, embeddings_model: str = "text-embedding-3-small",
                 llm_model: str = "gpt-5-mini", client: OpenAI | None = None):
        # Pass args up to RAGModel (base) class
        super().__init__(retriever, embeddings_model, llm_model, client)

    def retrieve(self, query_text: str, k: int) -> list[str]:
        query_emb = embed_query(query_text, self.embeddings_model, self.client)
        # TODO: test retrieval with mock embeddings to avoid wasting credits
        # self.retriever.query(query_emb, k)
        return ["Not implemented yet!"]

    def answer(self, query: str, k: int) -> str:
        docs = self.retrieve(query, k)
        return "Not implemented yet!"


