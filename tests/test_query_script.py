import pytest
from unittest.mock import patch, Mock

from scripts.query import build_parser, main


def test_query_parser():
    """
    Test argument parsing in `query.py` script.
    """
    parser = build_parser()

    # Minimal required arguments
    args_list = [
        "--query_text", "Why does my cat always wake me up in the middle of "
                        "the night?",
        "--index_path", "/fake-index",
        "--rag_model", "baseline"
    ]
    args = parser.parse_args(args_list)

    # Check required arguments
    assert args.query_text == ("Why does my cat always wake me up in the "
                               "middle of the night?")
    assert args.index_path == "/fake-index"
    assert args.rag_model == "baseline"

    # Check default values
    assert args.k == 5
    assert args.collection_name == "bright_documents_biology"
    assert args.embeddings_model == "text-embedding-3-small"
    assert args.llm_model == "gpt-5-mini"

    # Check types
    assert isinstance(args.index_path, str)
    assert isinstance(args.rag_model, str)
    assert isinstance(args.query_text, str)
    assert isinstance(args.k, int)

    # Check that overriding defaults works
    args_override = parser.parse_args([
        "--query_text", "Do AIs dream of electric sheep?",
        "--index_path", "/another-index",
        "--k", "25",
        "--rag_model", "improved",
        "--collection_name", "some_collection",
    ])
    assert args_override.query_text == "Do AIs dream of electric sheep?"
    assert args_override.index_path == "/another-index"
    assert args_override.k == 25
    assert args_override.collection_name == "some_collection"
    assert args_override.rag_model == "improved"

    # Check that missing required arguments raises SystemExit
    with pytest.raises(SystemExit):
        parser.parse_args(["--k", "5"])

    # Check choices enforcement
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--rag_model", "bad_model"
        ])

    # Check typing enforcement
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--k", "hello"
        ])

def test_simple_query():
    """
    Test LLM answer to a query using mocks.
    """
    fake_answer = "The powerhouse of the cell."

    with patch("scripts.query.chromadb.PersistentClient") as mock_chroma, \
         patch("scripts.query.OpenAI") as mock_openai, \
         patch("scripts.query.ImprovedRAG") as mock_improved:

        # Mock ChromaDB collection
        mock_chroma.return_value.get_collection.return_value = Mock()

        # Mock OpenAI client
        mock_openai.return_value = Mock()

        # Decide which RAG class is instantiated based on argv
        mock_rag_instance = mock_improved.return_value
        mock_rag_instance.answer.return_value = fake_answer

        argv = [
            "--query_text", "What is mitochondria?",
            "--index_path", "/fake-index",
            "--rag_model", "improved"
        ]
        result = main(argv)

        # Assertions
        mock_rag_instance.answer.assert_called_once_with(
            query_text="What is mitochondria?", k=5
        )
        assert result == fake_answer
