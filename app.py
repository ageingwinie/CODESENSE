"""Streamlit dashboard for CodeSense."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from search_engine import CodeSearchEngine


PROJECT_DIR = Path(__file__).resolve().parent
INDEX_PATH = PROJECT_DIR / "index_data.pkl"

st.set_page_config(page_title="CodeSense", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading CodeSense indexes and local embedding model...")
def get_engine(index_path: str) -> CodeSearchEngine:
    return CodeSearchEngine(index_path)


def score_summary(result: dict[str, object], mode: str) -> str:
    if mode == "Hybrid":
        return (
            f"Hybrid {result['hybrid_score']:.3f}  |  "
            f"Lexical {result['lexical_score']:.3f}  |  "
            f"AST {result['ast_score']:.3f}  |  "
            f"Semantic {result['semantic_score']:.3f}"
        )
    key = {"Lexical": "lexical_score", "AST Structural": "ast_score", "Semantic": "semantic_score"}[mode]
    return f"Score: {result[key]:.3f}"


def run_search(engine: CodeSearchEngine, query: str, mode: str, top_k: int) -> list[dict[str, object]]:
    if mode == "Lexical":
        return engine.text_search(query, top_k)
    if mode == "AST Structural":
        return engine.ast_structure_search(query, top_k)
    if mode == "Semantic":
        return engine.semantic_search(query, top_k)
    return engine.hybrid_search(query, top_k)


st.title("CodeSense")
st.caption("Advanced code snippet finder using AST structure, TF-IDF text matching, and local semantic retrieval.")

if not INDEX_PATH.exists():
    st.error("No search index is available yet.")
    st.code("python indexer.py --dataset sample_data.csv --output index_data.pkl", language="bash")
    st.stop()

try:
    engine = get_engine(str(INDEX_PATH))
except Exception as error:
    st.error(f"Could not load the index: {error}")
    st.stop()

with st.sidebar:
    st.header("Search settings")
    mode = st.selectbox("Retrieval mode", ["Hybrid", "Lexical", "AST Structural", "Semantic"])
    top_k = st.slider("Top results", min_value=1, max_value=min(10, len(engine.records)), value=min(5, len(engine.records)))
    st.caption(f"Indexed snippets: {len(engine.records)}")
    if engine.index.get("skipped_rows"):
        st.caption(f"Skipped malformed snippets: {len(engine.index['skipped_rows'])}")

query = st.text_input(
    "Describe the code you need",
    placeholder="e.g., loop through values and return only even numbers",
)

if query.strip():
    try:
        with st.spinner("Searching..."):
            results = run_search(engine, query, mode, top_k)
    except Exception as error:
        st.error(f"Search failed: {error}")
        st.stop()

    if mode == "AST Structural" and not any(result["ast_score"] > 0 for result in results):
        st.info("No structural terms were recognized. Try words such as loop, condition, function, exception, class, or list.")

    for result in results:
        with st.container(border=True):
            st.subheader(f"#{result['rank']} — {result['instruction']}")
            st.caption(score_summary(result, mode))
            st.code(result["code"], language="python", line_numbers=True)
            with st.expander("Structural details"):
                nodes = result["ast_nodes"]
                st.write("AST nodes: " + ", ".join(nodes[:60]) + (" …" if len(nodes) > 60 else ""))
                if result["docstrings"]:
                    st.write("Docstrings: " + " | ".join(result["docstrings"]))
                if result["comments"]:
                    st.write("Comments: " + " | ".join(result["comments"]))
else:
    st.info("Enter a natural-language request or a small Python fragment to search the local index.")
