# RAG improvements 📚⚡

## Table of contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Modeling approach](#modeling-approach)
4. [Usage](#usage)
5. [Evaluation results](#evaluation-results)
6. [Testing](#testing)
7. [Other features](#other-features)

## Overview  

This repository explores **retrieval-augmented generation (RAG)** with a focus on improving the retrieval stage through 
embedding-based techniques. The effectiveness of the improvements is quantified using several standard information 
retrieval metrics.  

All experiments rely on OpenAI API client, using the following models:  
- **`text-embedding-3-small`** for generating query embeddings.  
- **`gpt-5-mini`** as the large language model for generating responses.  

The document collection comes from a pre-indexed ChromaDB database built on a subset of 
[BRIGHT](https://huggingface.co/datasets/xlangai/BRIGHT), specifically the `biology` split. For evaluation, the 
`example` subset of the same split is used.  

The repository is organized into three subpackages:  
- **`models/`** contains the implemented RAG systems:  
  - `BaselineRAG`: a basic reference implementation.  
  - `ImprovedRAG`: a variant with pseudo-relevance feedback and query expansion.  
- **`scripts/`** contains scripts runnable from a CLI:  
  - `query.py`: runs direct RAG queries.  
  - `evaluation.py`: computes and prints evaluation results (see section below).
- **`utils/`** contains misc utilities, currently only the logging system.

## Installation

### Prerequisites
- Python 3.10 or higher
- ChromaDB index with the BRIGHT subset
- An OpenAI API key with access to the aforementioned models

1. Clone this repo and access its root directory

```bash
git clone https://github.com/ramcarreno/rag-improvements.git
cd rag-improvements
```

2. Create a virtual environment, activate it and install the dependencies in `pyproject.toml`. This can be simply done 
with:

```bash
python -m venv .venv
pip install .
```

Alternatively, you can use **uv**, so you don't need to worry about Python virtual environments! It's also faster :)

```bash
uv sync
```

3. Export your `OPENAI_API_KEY` as an environment variable and extract the ChromaDB index in a known location, 
preferably this same directory.

## Modeling approach

### Baseline: A classical RAG pipeline

The idea behind the *baseline model* is very simple: queries are embedded and fed to the retriever which returns the top 
`k` results, including the documents themselves and their ids for future evaluation. To generate LLM answers, a 
structured prompt fed with the query and retrieved context is sent to the LLM, previously indicating as a system 
instruction that the LLM in question is *an assistant designed to answer biology questions*.

### Improved: Pseudo-relevance feedback with query expansion

For the *improved model*, the main goal is to increase recall without changing the number of retrieved documents. 
I tried a few ideas:

- Splitting queries into segments and averaging embeddings sounded promising at first, hoping segments would give more 
direct signals to the retriever, but it ended up losing contextual info and hurt all metrics.
- Next, I tried a simple form of pseudo-relevance feedback (PRF): retrieve `m` initial documents, blend their embeddings
with the query embeddings with a weight factor, normalize, and then re-query. Recall@5 improved by a modest ~2% and 
other metrics remained stable.
- The version that really helped was PRF with query term expansion. I take the top `n` TF-IDF terms from the first 
retrieval with `m` documents, append them to the original query, generate new embeddings, and retrieve again. This gave 
a noticeable boost across most metrics, most increasing by 4–6 percentage points. 

This last method needs to call the embeddings API again initially, as the query text is modified, and also every time 
the `m` or `n` parameters are adjusted as a new textual query is created. Considering this, I implemented a simple 
embeddings caching mechanism that minimizes those calls ;)

A notable advantage of PRF is that it doesn't require slow and costly LLM paraphrasing, changing indexes, or modifying 
chunking. With cached embeddings, querying the entire test set can take roughly the same as using the baseline model
(around 3 seconds in my machine!)

### Notes on implementation

Both models inherit from an abstract base class which makes room for basic *retrieval* and *answer* methods, the first
being in charge of querying the vector database while the latter invokes the LLM. This ABC also implements batching and 
the caching solutions for query embeddings. Cached embeddings are stored in `.emb_cache/` as a 
[shelve](https://docs.python.org/3/library/shelve.html) which guarantees fast and easy access for this use case.

With respect to the two PRF approaches, I've kept the former as `retrieve_with_simple_prf` for experimental purposes. 
PRF hyperparameters such as `m`, `weight`, and TF-IDF `n` are changeable through each method but not at the CLI scripts 
level.

## Usage

**Note:** If you installed dependencies with `uv sync`, substitute `python` for `uv run` in the examples below.

### Running RAG queries

The query script produces a retrieval-informed LLM response to a query. Streaming is not enabled, so note LLM response
generation may take a few seconds.

```bash
python -m scripts.query \
    --query_text "Your biology question here" \
    --k 5 \
    --index_path /path/to/chromadb/index \
    --collection_name bright_documents_biology \
    --rag_model baseline \
    --embeddings_model text-embedding-3-small \
    --llm_model gpt-5-mini
```

#### Arguments

- `--query_text` *(str, required)* The query to run.
- `--k` *(int, default=5)* Number of documents to retrieve.
- `--index_path` *(str, required)* Path to the vector store index.
- `--collection_name` *(str, default="bright_documents_biology")* Name of the subset to select from the vector store.
- `--rag_model` *(str, choices=["baseline", "improved"])* RAG model to use.
- `--embeddings_model` *(str, default="text-embedding-3-small")* Model used to embed queries.
- `--llm_model` *(str, default="gpt-5-mini")* Model used to generate text responses.

### Obtaining evaluation results

The following script prints metrics for a given model, sorted by `k` since multiple values of `k` can be specified at 
once. Selected metrics are explained in [Evaluation results](#evaluation-results) section.

```bash
python -m scripts.evaluate \
    --k_values 5 10 \
    --index_path /path/to/chromadb/index \
    --collection_name bright_documents_biology \
    --hf_subset examples \
    --hf_split biology \
    --rag_model baseline \
    --embeddings_model text-embedding-3-small \
    --embedding_batch_size 32
```

#### Arguments

- `--k_values` *(list of int)* List of k values for @k evaluation metrics (e.g., `5 10`). Determines how many top 
documents are considered.
- `--index_path` *(str, required)* Path to the vector store index.
- `--collection_name` *(str, default="bright_documents_biology")* Name of the subset to select from the vector store.
- `--hf_subset` *(str, default="examples")* Hugging Face dataset subset to load for evaluation.
- `--hf_split` *(str, default="biology")* Dataset split to evaluate on.
- `--rag_model` *(str, choices=["baseline", "improved"])* RAG model to use.
- `--embeddings_model` *(str, default="text-embedding-3-small")* Embedding model for queries.
- `--embedding_batch_size` *(int, default=32)* Number of queries to embed at once when calling the embeddings API. Used
to pre-index testing dataset faster. Embeddings are also cached for future requests.

## Evaluation results

The selected metrics, at different retrieval (k) values, as previously introduced, and averaged over the entire 
`biology` split in the `example` test set were the following:

- **Recall**: The ratio of relevant documents retrieved across queries.
- **Hit**: The ratio of queries where at least one relevant document was retrieved.
- **Mean Reciprocal Rank (MRR)**: Takes retriever ranking into account. Average rank position of the first relevant 
document across queries.
- **Normalized Discounted Cumulative Gain (nDCG)**: Takes ranking into account as well, but measuring the quality of the
ranking of all relevant documents.

The original [BRIGHT](https://huggingface.co/datasets/xlangai/BRIGHT) dataset includes the retrieval gold labels in the
`gold_ids` column. Resulting metrics are reported below for k values of 5 and 10.

For `k=5`:

| Model                           | Recall@5 | Hit@5 | MRR@5 | nDCG@5 |
|---------------------------------|----------|-------|-------|--------|
| Baseline                        | 0.175    | 0.408 | 0.265 | 0.185  |
| Improved w/ simple PRF          | 0.192    | 0.408 | 0.270 | 0.197  |
| Improved w/ PRF query expansion | 0.233    | 0.466 | 0.353 | 0.256  |

For `k=10`:

| Model                           | Recall@10 | Hit@10 | MRR@10 | nDCG@10 |
|---------------------------------|-----------|--------|--------|---------|
| Baseline                        | 0.267     | 0.495  | 0.276  | 0.213   |
| Improved w/ simple PRF          | 0.269     | 0.485  | 0.279  | 0.218   |
| Improved w/ PRF query expansion | 0.284     | 0.495  | 0.357  | 0.265   |

For the simple PRF hyperparameters, I found best results setting `prf_m=3` for the first retrieval step and 
`prf_weight=0.1` applied to the pseudo-relevant retrieved documents. In the case of query expansion `prf_m=10` and 
TF-IDF top terms `prf_n_terms=50` for these results, although there might be better combinations. Remember these 
parameters must be changed directly inside each method.

## Testing

Work in progress...

## Other features

Work in progress...