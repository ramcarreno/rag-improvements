from unittest.mock import MagicMock

from models.baseline import BaselineRAG


def test_retrieval():
    """
    Test retrieval with BaselineRAG and mock embeddings.
    """
    # Mock retriever
    mock_retriever = MagicMock()
    fake_result = {
        "documents": ["doc1", "doc2"],
        "ids": [["id1", "id2"]],
        "distances": [[0.1, 0.2]]
    }
    mock_retriever.query.return_value = fake_result

    # Instantiate BaselineRAG with the mock retriever
    rag_model = BaselineRAG(retriever=mock_retriever)

    # Set logger from ABC to avoid exception
    mock_logger = MagicMock()
    rag_model.logger = mock_logger

    # Patch embed_query to return a fake embedding
    rag_model.embed_query = MagicMock(return_value=[0.1, 0.2, 0.3])

    # Call retrieve
    result = rag_model.retrieve("test query", k=2)

    # Assertions
    mock_retriever.query.assert_called_once_with(
        query_embeddings=[[0.1, 0.2, 0.3]],
        n_results=2
    )
    assert result == fake_result
    assert "documents" in result
    assert "ids" in result
    assert len(result["ids"][0]) == 2
    assert result["ids"][0] == ["id1", "id2"]
