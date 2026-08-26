"""Query the persisted CodeSense hybrid indexes."""

from __future__ import annotations

import ast
import pickle
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from indexer import DEFAULT_MODEL, preprocess_text


AST_KEYWORDS: dict[str, tuple[str, ...]] = {
    "function": ("FunctionDef",),
    "method": ("FunctionDef",),
    "argument": ("arguments",),
    "parameter": ("arguments",),
    "return": ("Return",),
    "loop": ("For", "While"),
    "iterate": ("For",),
    "iteration": ("For",),
    "for": ("For",),
    "while": ("While",),
    "condition": ("If",),
    "conditional": ("If",),
    "branch": ("If",),
    "if": ("If",),
    "else": ("If",),
    "list": ("List",),
    "dictionary": ("Dict",),
    "dict": ("Dict",),
    "set": ("Set",),
    "tuple": ("Tuple",),
    "class": ("ClassDef",),
    "exception": ("Try", "ExceptHandler"),
    "error": ("Try", "ExceptHandler"),
    "try": ("Try",),
    "file": ("With", "Call"),
    "context": ("With",),
    "comprehension": ("ListComp", "DictComp", "SetComp", "GeneratorExp"),
    "lambda": ("Lambda",),
    "recursive": ("Call", "FunctionDef"),
    "recursion": ("Call", "FunctionDef"),
    "add": ("BinOp",),
    "sum": ("BinOp",),
    "multiply": ("BinOp",),
    "comparison": ("Compare",),
    "compare": ("Compare",),
    "boolean": ("BoolOp",),
    "import": ("Import", "ImportFrom"),
    "async": ("AsyncFunctionDef", "Await"),
}
WORD_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_]*")


def extract_query_ast_features(query: str) -> list[str]:
    """Extract AST structure from code queries, or infer structure from natural language."""
    features: list[str] = []
    try:
        tree = ast.parse(query)
        features.extend(type(node).__name__ for node in ast.walk(tree) if not isinstance(node, ast.Module))
    except (SyntaxError, ValueError, TypeError):
        pass

    words = set(WORD_PATTERN.findall(query.lower()))
    for word in words:
        features.extend(AST_KEYWORDS.get(word, ()))
    return features


class CodeSearchEngine:
    """CPU-friendly hybrid search over a serialized CodeSense index."""

    def __init__(self, index_path: str | Path = "index_data.pkl", model_name: str | None = None):
        path = Path(index_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Index not found at {path}. Run `python indexer.py --dataset sample_data.csv` first."
            )
        with path.open("rb") as handle:
            self.index: dict[str, Any] = pickle.load(handle)
        self.records: list[dict[str, Any]] = self.index["records"]
        self.vectorizer = self.index["tfidf_vectorizer"]
        self.tfidf_matrix = self.index["tfidf_matrix"]
        self.embeddings = np.asarray(self.index["embeddings"], dtype=np.float32)
        self.model_name = model_name or self.index.get("model_name", DEFAULT_MODEL)
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    @staticmethod
    def _min_max(scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        if scores.size == 0:
            return scores
        minimum, maximum = float(scores.min()), float(scores.max())
        if np.isclose(minimum, maximum):
            return np.ones_like(scores) if maximum > 0 else np.zeros_like(scores)
        return (scores - minimum) / (maximum - minimum)

    @staticmethod
    def _limit(results: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        return results[: max(1, int(top_k))]

    def _result_rows(self, scores: np.ndarray, score_key: str) -> list[dict[str, Any]]:
        order = np.argsort(-scores, kind="stable")
        rows: list[dict[str, Any]] = []
        for rank, index in enumerate(order, start=1):
            record = self.records[int(index)]
            rows.append(
                {
                    "rank": rank,
                    "id": record["id"],
                    "instruction": record["instruction"],
                    "code": record["code"],
                    "docstrings": record["docstrings"],
                    "comments": record["comments"],
                    "ast_nodes": record["ast_nodes"],
                    score_key: float(scores[index]),
                }
            )
        return rows

    def _lexical_scores(self, query: str) -> np.ndarray:
        query_vector = self.vectorizer.transform([preprocess_text(query)])
        return (self.tfidf_matrix @ query_vector.T).toarray().ravel()

    def _ast_scores(self, query: str) -> np.ndarray:
        query_nodes = extract_query_ast_features(query)
        if not query_nodes:
            return np.zeros(len(self.records), dtype=float)
        query_counts = Counter(query_nodes)
        scores: list[float] = []
        for record in self.records:
            candidate_counts = Counter(record["ast_counts"])
            overlap = sum(min(query_counts[node], candidate_counts[node]) for node in query_counts)
            union = sum(max(query_counts[node], candidate_counts[node]) for node in set(query_counts) | set(candidate_counts))
            jaccard = overlap / union if union else 0.0
            sequence_similarity = SequenceMatcher(None, query_nodes, record["ast_nodes"]).ratio()
            scores.append(0.8 * jaccard + 0.2 * sequence_similarity)
        return np.asarray(scores, dtype=float)

    def _semantic_scores(self, query: str) -> np.ndarray:
        query_embedding = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )[0]
        return self.embeddings @ np.asarray(query_embedding, dtype=np.float32)

    def text_search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Rank snippets by TF-IDF similarity over normalized English instructions."""
        return self._limit(self._result_rows(self._lexical_scores(query), "lexical_score"), top_k)

    def ast_structure_search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Rank snippets by AST node-count overlap plus node-sequence similarity."""
        return self._limit(self._result_rows(self._ast_scores(query), "ast_score"), top_k)

    def semantic_search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Rank snippets by cosine similarity in sentence-transformer embedding space."""
        return self._limit(self._result_rows(self._semantic_scores(query), "semantic_score"), top_k)

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        lexical_weight: float = 0.35,
        ast_weight: float = 0.25,
        semantic_weight: float = 0.40,
    ) -> list[dict[str, Any]]:
        """Fuse min-max-normalized lexical, structural, and semantic retrieval scores."""
        weights = np.asarray([lexical_weight, ast_weight, semantic_weight], dtype=float)
        if np.any(weights < 0) or np.isclose(weights.sum(), 0):
            raise ValueError("Hybrid weights must be non-negative and sum to a positive value.")
        weights /= weights.sum()

        lexical = self._lexical_scores(query)
        structural = self._ast_scores(query)
        semantic = self._semantic_scores(query)
        fused = (
            weights[0] * self._min_max(lexical)
            + weights[1] * self._min_max(structural)
            + weights[2] * self._min_max(semantic)
        )
        order = np.argsort(-fused, kind="stable")
        results: list[dict[str, Any]] = []
        for rank, index in enumerate(order, start=1):
            record = self.records[int(index)]
            results.append(
                {
                    "rank": rank,
                    "id": record["id"],
                    "instruction": record["instruction"],
                    "code": record["code"],
                    "docstrings": record["docstrings"],
                    "comments": record["comments"],
                    "ast_nodes": record["ast_nodes"],
                    "lexical_score": float(lexical[index]),
                    "ast_score": float(structural[index]),
                    "semantic_score": float(semantic[index]),
                    "hybrid_score": float(fused[index]),
                }
            )
        return self._limit(results, top_k)

