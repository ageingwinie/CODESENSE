# CodeSense

CodeSense searches Python snippets with three complementary local signals:

- **Lexical:** preprocessed English instructions and TF-IDF.
- **Structural:** Python `ast` node sequences and node-count overlap.
- **Semantic:** CPU sentence-transformer embeddings of instructions, docstrings, and comments.

## Setup

Use Python 3.10+ and create an isolated environment if desired:

```powershell
cd C:\Users\ARJUN\Documents\Codex\2026-08-18\i-w\outputs\CodeSense
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Build the local index from the included data:

```powershell
python indexer.py --dataset sample_data.csv --output index_data.pkl
```

Then launch the dashboard:

```powershell
streamlit run app.py
```

The first indexing/search run downloads the public `all-MiniLM-L6-v2` model into the Sentence Transformers cache. Afterwards, normal indexing and searching run locally on CPU with no API key. To operate in an air-gapped environment, pre-download the model once and pass its local directory with `--model C:\path\to\all-MiniLM-L6-v2` when building the index; the application stores and reuses that path.

## Dataset format

Supply CSV, JSON, JSONL, or NDJSON with a code field named one of `code`, `snippet`, `source`, or `python_code`; and a text field named one of `instruction`, `docstring`, `description`, `text`, or `query`. Invalid Python snippets are recorded and skipped without stopping indexing.
