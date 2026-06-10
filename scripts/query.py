#!/usr/bin/env python3
"""Interactive query interface for the Horatio neuro-symbolic pipeline.

Accepts natural language questions, retrieves the most relevant chunks
from the ChromaDB vector store, and uses Claude to synthesize a cited
answer grounded strictly in the retrieved content.

The LLM synthesis step is explicitly constrained: Claude may only use
the retrieved chunks as source material and must cite the source URL
and heading for every claim it makes.  If the retrieved chunks do not
contain enough information to answer the question, Claude is instructed
to say so rather than speculate.

Usage:
    python3 query.py                        # interactive REPL
    python3 query.py --query "your question"  # single query, then exit
    python3 query.py --no-synthesis         # retrieval only, skip LLM step
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import anthropic
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_DB_DIR = Path("data/horatio/chroma")
DEFAULT_COLLECTION = "scc-docs"
DEFAULT_EMBEDDING_MODEL = "multi-qa-mpnet-base-dot-v1"
DEFAULT_TOP_K = 5
DEFAULT_MAX_TOKENS = 1024

# ---------------------------------------------------------------------------
# Synthesis prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a technical documentation assistant for Cisco Security Cloud Control.
Your answers are based exclusively on retrieved documentation chunks provided
to you in each query.  You must follow these rules without exception:

1. Only use information present in the provided chunks.  Do not add, infer,
   or speculate beyond what the chunks explicitly state.

2. Cite your sources.  For every substantive claim, include a citation in
   the format: [Source: <page title>, <heading path>].  Include the URL on
   a separate line after the answer as a reference list.

3. If the chunks do not contain enough information to answer the question,
   say so clearly and specifically: state what the question asks and what
   is missing.  Do not attempt to answer from general knowledge.

4. If chunks contain partially relevant information, use what is relevant
   and note what aspects of the question remain unanswered.

5. Be concise.  Prefer direct answers over lengthy preamble.
"""

USER_PROMPT_TEMPLATE = """\
Question: {question}

Retrieved documentation chunks:

{chunks}

Answer the question using only the chunks above.  Cite every claim.
"""

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(
    collection: chromadb.Collection,
    question: str,
    top_k: int,
) -> list[dict]:
    """Query ChromaDB and return the top-k chunks with metadata."""
    results = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "body": doc,
            "distance": dist,
            "source_url": meta.get("source_url", ""),
            "source_title": meta.get("source_title", ""),
            "heading_text": meta.get("heading_text", ""),
            "heading_path": meta.get("heading_path", ""),
            "content_type": meta.get("content_type", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "chunk_total": meta.get("chunk_total", 1),
        })
    return chunks


def format_chunks_for_prompt(chunks: list[dict]) -> str:
    """Format retrieved chunks into a prompt-ready string."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        path = chunk["heading_path"]
        heading = chunk["heading_text"]
        full_path = f"{path} | {heading}" if path else heading
        chunk_note = (
            f" (part {chunk['chunk_index'] + 1} of {chunk['chunk_total']})"
            if chunk["chunk_total"] > 1
            else ""
        )
        parts.append(
            f"--- Chunk {i}{chunk_note} ---\n"
            f"Page: {chunk['source_title']}\n"
            f"Section: {full_path}\n"
            f"URL: {chunk['source_url']}\n"
            f"Content:\n{chunk['body']}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def synthesize(
    client: anthropic.Anthropic,
    question: str,
    chunks: list[dict],
    model: str,
    max_tokens: int,
) -> str:
    """Send retrieved chunks to Claude and return a cited answer."""
    formatted_chunks = format_chunks_for_prompt(chunks)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        question=question,
        chunks=formatted_chunks,
    )
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_retrieval(chunks: list[dict]) -> None:
    """Print retrieved chunks in a readable format."""
    print(f"\n{'─' * 60}")
    print(f"Retrieved {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, start=1):
        path = chunk["heading_path"]
        heading = chunk["heading_text"]
        full_path = f"{path} | {heading}" if path else heading
        print(f"\n  {i}. [{chunk['distance']:.4f}] {full_path}")
        print(f"     {chunk['source_title']} — {chunk['source_url']}")
        print(f"     {chunk['body'][:100]}...")


def display_answer(answer: str) -> None:
    """Print the synthesized answer."""
    print(f"\n{'─' * 60}")
    print("Answer:\n")
    print(answer)
    print(f"{'─' * 60}")


# ---------------------------------------------------------------------------
# Query loop
# ---------------------------------------------------------------------------

def run_query(
    question: str,
    collection: chromadb.Collection,
    anthropic_client: anthropic.Anthropic | None,
    top_k: int,
    model: str,
    max_tokens: int,
    show_retrieval: bool,
    no_synthesis: bool,
) -> None:
    """Execute one query: retrieve, optionally display, optionally synthesize."""
    chunks = retrieve(collection, question, top_k)

    if show_retrieval or no_synthesis:
        display_retrieval(chunks)

    if no_synthesis or anthropic_client is None:
        return

    answer = synthesize(anthropic_client, question, chunks, model, max_tokens)
    display_answer(answer)


def run_repl(
    collection: chromadb.Collection,
    anthropic_client: anthropic.Anthropic | None,
    top_k: int,
    model: str,
    max_tokens: int,
    show_retrieval: bool,
    no_synthesis: bool,
) -> None:
    """Run an interactive query loop until the user exits."""
    print("\nHoratio query interface")
    print(f"Collection: {collection.name}  |  Top-k: {top_k}  |  Model: {model}")
    print("Type a question and press Enter.  Type 'exit' or Ctrl-C to quit.\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Exiting.")
            break

        run_query(
            question=question,
            collection=collection,
            anthropic_client=anthropic_client,
            top_k=top_k,
            model=model,
            max_tokens=max_tokens,
            show_retrieval=show_retrieval,
            no_synthesis=no_synthesis,
        )
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive query interface for the Horatio neuro-symbolic pipeline. "
            "Retrieves relevant documentation chunks and synthesizes cited answers "
            "using Claude."
        )
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Run a single query and exit (omit for interactive REPL mode).",
    )
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=DEFAULT_DB_DIR,
        help=f"ChromaDB directory (default: {DEFAULT_DB_DIR})",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION,
        help=f"ChromaDB collection name (default: {DEFAULT_COLLECTION})",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Sentence-transformers model for query embedding (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Claude model for answer synthesis (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of chunks to retrieve (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum tokens in Claude's response (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--show-retrieval",
        action="store_true",
        help="Print retrieved chunks before the synthesized answer.",
    )
    parser.add_argument(
        "--no-synthesis",
        action="store_true",
        help="Skip the LLM synthesis step and show only retrieved chunks.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    # --- Anthropic client ---
    if not args.no_synthesis:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print(
                "Error: ANTHROPIC_API_KEY environment variable is not set.\n"
                "Set it with: set -Ux ANTHROPIC_API_KEY sk-ant-...\n"
                "Or run with --no-synthesis to skip the LLM step.",
                file=sys.stderr,
            )
            return 2
        anthropic_client = anthropic.Anthropic(api_key=api_key)
    else:
        anthropic_client = None

    # --- ChromaDB ---
    if not args.db_dir.exists():
        print(
            f"Error: ChromaDB directory not found: {args.db_dir}\n"
            "Run embed-chunks.py first to populate the vector store.",
            file=sys.stderr,
        )
        return 2

    print(f"Loading embedding model: {args.embedding_model}", flush=True)
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=args.embedding_model,
        device="cpu",
        normalize_embeddings=True,
    )

    client = chromadb.PersistentClient(path=str(args.db_dir))
    try:
        collection = client.get_collection(
            name=args.collection,
            embedding_function=embedding_fn,
        )
    except Exception:
        print(
            f"Error: Collection '{args.collection}' not found in {args.db_dir}.\n"
            "Run embed-chunks.py first to populate the vector store.",
            file=sys.stderr,
        )
        return 2

    print(f"Collection '{args.collection}': {collection.count()} documents.")

    # --- Single query or REPL ---
    if args.query:
        run_query(
            question=args.query,
            collection=collection,
            anthropic_client=anthropic_client,
            top_k=args.top_k,
            model=args.model,
            max_tokens=args.max_tokens,
            show_retrieval=args.show_retrieval,
            no_synthesis=args.no_synthesis,
        )
    else:
        run_repl(
            collection=collection,
            anthropic_client=anthropic_client,
            top_k=args.top_k,
            model=args.model,
            max_tokens=args.max_tokens,
            show_retrieval=args.show_retrieval,
            no_synthesis=args.no_synthesis,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())