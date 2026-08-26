"""Build persistent lexical, structural, and semantic indexes for CodeSense.

Usage:
    python indexer.py --dataset sample_data.csv --output index_data.pkl
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import pickle
import re
import tokenize
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nltk.stem import SnowballStemmer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


DEFAULT_MODEL = "all-MiniLM-L6-v2"
TEXT_COLUMN_ALIASES = ("instruction", "docstring", "description", "text", "query")
CODE_COLUMN_ALIASES = ("code", "snippet", "source", "python_code")
TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_]{1,}")
STEMMER = SnowballStemmer("english")


def preprocess_text(text: object) -> str:
    """Lowercase, tokenize, remove English stop words, and stem free-form text."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    tokens = TOKEN_PATTERN.findall(str(text).lower())
    return " ".join(STEMMER.stem(token) for token in tokens if token not in ENGLISH_STOP_WORDS)


def _find_column(columns: list[str], aliases: tuple[str, ...], purpose: str) -> str:
    normalized = {column.strip().lower(): column for column in columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    raise ValueError(
        f"Dataset needs a {purpose} column. Accepted names: {', '.join(aliases)}. "
        f"Found: {', '.join(columns)}"
    )


def load_dataset(dataset_path: str | Path) -> pd.DataFrame:
    """Load CSV, JSON, or JSONL data and normalize it to code/instruction columns."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix == ".json":
        try:
            frame = pd.read_json(path)
        except ValueError:
            frame = pd.read_json(path, lines=True)
    elif suffix in {".jsonl", ".ndjson"}:
        frame = pd.read_json(path, lines=True)
    else:
        raise ValueError("Supported dataset formats are CSV, JSON, JSONL, and NDJSON.")

    code_column = _find_column(frame.columns.tolist(), CODE_COLUMN_ALIASES, "code")
    text_column = _find_column(frame.columns.tolist(), TEXT_COLUMN_ALIASES, "instruction/docstring")
    cleaned = frame[[code_column, text_column]].rename(
        columns={code_column: "code", text_column: "instruction"}
    )
    cleaned = cleaned.dropna(subset=["code", "instruction"])
    cleaned["code"] = cleaned["code"].astype(str)
    cleaned["instruction"] = cleaned["instruction"].astype(str)
    return cleaned


def extract_comments(source: str) -> list[str]:
    """Return comment bodies where possible; malformed source simply yields no comments."""
    comments: list[str] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                comments.append(token.string.lstrip("#").strip())
    except (tokenize.TokenError, IndentationError):
        pass
    return comments


def extract_ast_data(source: str) -> tuple[list[str], dict[str, int], list[str]]:
    """Parse source into preorder AST node names, counts, and discovered docstrings.

    Syntax errors are intentionally re-raised so the caller can skip only that row.
    """
    tree = ast.parse(source)
    node_sequence = [type(node).__name__ for node in ast.walk(tree)]
    docstrings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            docstring = ast.get_docstring(node, clean=True)
            if docstring:
                docstrings.append(docstring)
    return node_sequence, dict(Counter(node_sequence)), docstrings


def semantic_document(instruction: str, docstrings: list[str], comments: list[str]) -> str:
    """Create a code-aware description without embedding raw identifiers as the main signal."""
    parts = [instruction.strip(), *docstrings, *comments]
    return "\n".join(part.strip() for part in parts if part and part.strip())


def build_index(dataset_path: str | Path, output_path: str | Path, model_name: str) -> dict[str, Any]:
    """Construct all three indexes and write a portable pickle artifact."""
    source_rows = load_dataset(dataset_path)
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for original_row, row in source_rows.reset_index(drop=True).iterrows():
        code = row["code"]
        try:
            ast_nodes, ast_counts, docstrings = extract_ast_data(code)
        except (SyntaxError, ValueError, TypeError) as error:
            skipped.append({"row": str(original_row), "reason": str(error)})
            continue

        comments = extract_comments(code)
        instruction = row["instruction"].strip()
        records.append(
            {
                "id": len(records),
                "source_row": int(original_row),
                "code": code,
                "instruction": instruction,
                "docstrings": docstrings,
                "comments": comments,
                "semantic_text": semantic_document(instruction, docstrings, comments),
                "ast_nodes": ast_nodes,
                "ast_counts": ast_counts,
            }
        )

    if not records:
        raise ValueError("No valid Python snippets remained after parsing the dataset.")

    lexical_documents = [preprocess_text(record["instruction"]) for record in records]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(lexical_documents)

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name, device="cpu")
    embedding_texts = [record["semantic_text"] for record in records]
    embeddings = model.encode(
        embedding_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    artifact: dict[str, Any] = {
        "version": 1,
        "model_name": model_name,
        "records": records,
        "tfidf_vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "embeddings": embeddings,
        "skipped_rows": skipped,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Indexed {len(records)} valid snippets; skipped {len(skipped)} malformed snippets.")
    print(f"Saved index to {target.resolve()}")
    if skipped:
        print("Skipped-row details:")
        print(json.dumps(skipped, indent=2))
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CodeSense search indexes.")
    parser.add_argument("--dataset", default="sample_data.csv", help="CSV, JSON, or JSONL dataset path")
    parser.add_argument("--output", default="index_data.pkl", help="Pickle index output path")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="SentenceTransformer model name or local model directory",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    build_index(arguments.dataset, arguments.output, arguments.model)
