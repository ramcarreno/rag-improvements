from openai import OpenAI


def embed_query(query_text: str, embeddings_model: str, client: OpenAI | None):
    response = client.embeddings.create(
        input=query_text,
        model=embeddings_model
    )
    # Extract and return the embedding vector
    return response.data[0].embedding
