import pytest
from scripts.evaluate import build_parser


def test_eval_parser():
    """
    Test argument parsing in `evaluate.py` script.
    """
    parser = build_parser()

    # Minimal required arguments
    args_list = [
        "--index_path", "/fake-index",
        "--rag_model", "baseline"
    ]
    args = parser.parse_args(args_list)

    # Check required arguments
    assert args.index_path == "/fake-index"
    assert args.rag_model == "baseline"

    # Check default values
    assert args.k_values == [5, 10]
    assert args.collection_name == "bright_documents_biology"
    assert args.hf_subset == "examples"
    assert args.hf_split == "biology"
    assert args.embeddings_model == "text-embedding-3-small"
    assert args.embedding_batch_size == 32

    # Check types
    assert isinstance(args.index_path, str)
    assert isinstance(args.rag_model, str)
    assert isinstance(args.k_values, list)
    assert all(isinstance(k, int) for k in args.k_values)

    # Check that overriding defaults works
    args_override = parser.parse_args([
        "--index_path", "/another-index",
        "--k_values", "5", "10", "20", "50",
        "--rag_model", "improved",
        "--collection_name", "some_collection",
    ])
    assert args_override.index_path == "/another-index"
    assert args_override.k_values == [5, 10, 20, 50]
    assert args_override.collection_name == "some_collection"
    assert args_override.rag_model == "improved"

    # Check that missing required arguments raises SystemExit
    with pytest.raises(SystemExit):
        parser.parse_args(["--k_values", "5"])

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