from openai import OpenAI


def embed_query(query_text: str, embeddings_model: str, client: OpenAI | None) \
        -> list[float]:
    """
    Generate OpenAI embeddings for a RAG query.

    Args:
        query_text (str): RAG query in text format.
        embeddings_model (str): OpenAI embeddings model name.
        client (OpenAI | None): OpenAI client.

    Returns:
        list[float]: Text embedding vector.
    """
    response = client.embeddings.create(
        input=query_text,
        model=embeddings_model
    )
    # Extract and return the embedding vector
    return response.data[0].embedding
