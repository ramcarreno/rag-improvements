import numpy as np
from typing import Any

from openai import OpenAI
from models.base import RAGModel
from sklearn.feature_extraction.text import TfidfVectorizer



class ImprovedRAG(RAGModel):
    """
    Improved RAG model using PRF techniques.
    """
    def __init__(self, retriever: Any,
                 embeddings_model: str = "text-embedding-3-small",
                 llm_model: str | None = None,
                 client: OpenAI | None = None):
        super().__init__(retriever, embeddings_model, llm_model, client)

    def retrieve(self, query_text: str, k: int) -> dict[str, list]:
        """
        Retrieval with TF-IDF query expansion pseudo-relevance feedback (PRF).
        """
        # PRF hyperparameters
        prf_m: int = 10  # pseudo-relevant docs
        prf_n_terms: int = 50  # n of tf-idf terms

        # Embed query & first retrieval, extract docs
        try:
            first_pass: dict[str, Any] = self.retriever.query(
                query_embeddings=[self.embed_query(query_text)],
                n_results=max(k, prf_m)
            )
        except Exception:
            self.logger.exception("[PRF first pass] Failed to retrieve "
                                  "documents!")
            raise
        top_docs: list[str] = first_pass["documents"][0][:prf_m]

        # Extract top TF-IDF terms from top m docs and expand query with them
        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                max_features=prf_n_terms
            )
            vectorizer.fit(top_docs)
            top_terms = vectorizer.get_feature_names_out()
        except ValueError:
            self.logger.exception("TF-IDF vectorizer failed while extracting "
                                  "top terms")
            raise
        expanded_query_text: str = query_text + " " + " ".join(top_terms)

        # Embed expanded query
        query_exp_emb = np.array(self.embed_query(expanded_query_text))
        query_exp_emb = query_exp_emb / np.linalg.norm(query_exp_emb)

        # Retrieve using this expanded embedding
        try:
            second_pass: dict[str, Any] = self.retriever.query(
                query_embeddings=[query_exp_emb.tolist()],
                n_results=k
            )
        except Exception:
            self.logger.exception("[PRF second pass] Failed to retrieve "
                                  "documents!")
            raise
        self.logger.info(f"Retrieved docs ids: {second_pass['ids'][0]}")

        return second_pass

    def retrieve_with_simple_prf(self, query_text: str, k: int) \
            -> dict[str, list]:
        """
        Retrieval with simple pseudo-relevance feedback (PRF).
        """
        # PRF hyperparameters
        prf_m: int = 5  # pseudo-relevant docs
        prf_weight: float = 0.1  # weight of m docs

        # Embed query & first retrieval, expose embeddings
        query_embeddings: list[float] = self.embed_query(query_text)
        try:
            first_pass: dict[str, Any] = self.retriever.query(
                query_embeddings=[query_embeddings],
                n_results=max(k, prf_m),
                include=["documents", "embeddings"]
            )
        except Exception:
            self.logger.exception("[PRF first pass] Failed to retrieve "
                                  "documents!")
            raise

        # Collect embeddings of top m docs and average them
        feedback_emb = np.mean(np.stack(first_pass["embeddings"][0]), axis=0)

        # Blend query according to weight and normalize
        query_prf_emb = ((1 - prf_weight) * np.array(query_embeddings)
                         + prf_weight * feedback_emb)
        query_prf_emb = query_prf_emb / np.linalg.norm(query_prf_emb)

        # Retrieve using informed embedding
        try:
            second_pass: dict[str, Any] = self.retriever.query(
                query_embeddings=[query_prf_emb.tolist()],
                n_results=k
            )
        except Exception:
            self.logger.exception("[PRF second pass] Failed to retrieve "
                                  "documents!")
            raise
        self.logger.info(f"Retrieved docs ids: {second_pass['ids'][0]}")

        return second_pass

    def answer(self, query_text: str, k: int) -> str:
        """
        Answers the query at LLM level through a simple (but slightly more
        nuanced) structured prompt.
        """
        # Return k most relevant documents
        result: dict[str, Any] = self.retrieve(query_text, k)
        documents: list[str] = result.get("documents", [[]])[0]

        # Build context and prompt from retrieval result
        rag_context: str = "\n\n".join(documents)
        user_prompt: str = (
            f"Context:\n{rag_context}\n\n"
            f"Question: {query_text}\n\n"
            "Answer clearly and concisely based only on the context "
            "provided. If multiple pieces of evidence support your answer, "
            "synthesize them."
        )

        # Call OpenAI API and return text response
        try:
            response = self.client.responses.create(
                model=self.llm_model,
                input=[
                    {"role": "system", "content": "You are a helpful assistant "
                                                  "answering biology questions."},
                    {"role": "user", "content": user_prompt}
                ]
            )
        except Exception:
            self.logger.exception("Failed to generate an LLM answer!")
            raise

        return response.output_text