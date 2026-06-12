#!/usr/bin/env python3
"""Hybrid query interface for the Horatio neuro-symbolic pipeline.

Combines symbolic graph retrieval (SPARQL over RDFLib) with semantic
vector search (ChromaDB) to deliver accurate, cited answers grounded
strictly in the documentation corpus.

Query flow
──────────
1. Concept detection — scan the question for known products, roles, and
   features using the same term lists used during graph construction.

2. Graph pre-filtering (symbolic) — if concepts are detected, query the
   RDF graph via SPARQL to find all pages that mention those concepts.
   This produces a candidate URL set that scopes the vector search.
   If no concepts are detected the system falls back to unscoped search.

3. Scoped vector search (semantic) — query ChromaDB for the top-k most
   similar chunks, filtered to the candidate URL set when available.
   Filtering is applied via ChromaDB's metadata `where` clause.

4. LLM synthesis — pass retrieved chunks to Claude with a strict
   grounding prompt: cite every claim, do not speculate beyond the chunks.

The graph layer never blocks a query.  It only narrows the search space
when it recognises named concepts, improving precision without risking
regression on general questions.

Usage
─────
    python3 query-hybrid.py                         # interactive REPL
    python3 query-hybrid.py -q "your question"      # single query, then exit
    python3 query-hybrid.py --no-synthesis          # retrieval only, skip LLM
    python3 query-hybrid.py --no-graph              # vector-only (baseline mode)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import anthropic
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rdflib import Graph

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_LLM_MODEL        = "claude-haiku-4-5"
DEFAULT_DB_DIR           = Path("data/horatio/chroma")
DEFAULT_GRAPH_FILE       = Path("data/horatio/graph.ttl")
DEFAULT_COLLECTION       = "scc-docs"
DEFAULT_EMBEDDING_MODEL  = "multi-qa-mpnet-base-dot-v1"
DEFAULT_TOP_K            = 5
DEFAULT_MAX_TOKENS       = 1024

# ---------------------------------------------------------------------------
# Known concept term lists
# Must match build-graph.py exactly so detection aligns with graph content.
# ---------------------------------------------------------------------------

KNOWN_PRODUCTS: list[tuple[str, list[str]]] = [
    ("AI Defense",                  ["AI Defense"]),
    ("Firewall Management",         ["Firewall Management", "FMC"]),
    ("Multicloud Defense",          ["Multicloud Defense"]),
    ("Secure Access",               ["Secure Access"]),
    ("Secure Workload",             ["Secure Workload"]),
    ("Secure Email Threat Defense", ["Secure Email Threat Defense"]),
    ("Secure Endpoint",             ["Secure Endpoint"]),
    ("Cisco Duo",                   ["Cisco Duo", "Duo"]),
    ("Cisco XDR",                   ["Cisco XDR", "XDR"]),
    ("SASE Management",             ["SASE Management", "SASE", "Secure Access Service Edge"]),
    ("SD-WAN",                      ["SD-WAN", "Catalyst SD-WAN"]),
]

KNOWN_ROLES: list[tuple[str, list[str]]] = [
    ("Organization Administrator",  ["organization administrator", "org admin"]),
    ("Super Admin",                 ["super admin", "SuperAdmin"]),
    ("Read Only User",              ["read only", "read-only user"]),
]

KNOWN_FEATURES: list[tuple[str, list[str]]] = [
    ("Role-Based Access Control",   ["role-based access control", "RBAC"]),
    ("Platform Management",         ["Platform Management"]),
    ("AI Assistant",                ["AI Assistant", "Cisco AI Assistant"]),
    ("Global Search",               ["Global Search"]),
    ("Shared Objects",              ["Shared Objects"]),
    ("Multi-Org",                   ["Multi-Org", "multi-org", "multi-organization"]),
    ("Claim Code",                  ["claim code"]),
    ("Identity Provider",           ["identity provider", "IdP", "SAML"]),
    ("Single Sign-On",              ["single sign-on", "SSO", "Security Cloud Sign On"]),
]

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
# Concept detection
# ---------------------------------------------------------------------------

def build_concept_patterns() -> list[tuple[re.Pattern, str, str]]:
    """Compile regex patterns for all known concepts."""
    patterns = []
    for label, aliases in KNOWN_PRODUCTS:
        for alias in aliases:
            patterns.append((
                re.compile(r'\b' + re.escape(alias) + r'\b'),
                "Product", label,
            ))
    for label, aliases in KNOWN_ROLES:
        for alias in aliases:
            patterns.append((
                re.compile(r'\b' + re.escape(alias) + r'\b', re.IGNORECASE),
                "Role", label,
            ))
    for label, aliases in KNOWN_FEATURES:
        for alias in aliases:
            patterns.append((
                re.compile(r'\b' + re.escape(alias) + r'\b', re.IGNORECASE),
                "Feature", label,
            ))
    return patterns


CONCEPT_PATTERNS = build_concept_patterns()


def detect_concepts(question: str) -> list[tuple[str, str]]:
    """Return list of (concept_type, canonical_label) found in the question."""
    found: set[tuple[str, str]] = set()
    for pattern, concept_type, label in CONCEPT_PATTERNS:
        if pattern.search(question):
            found.add((concept_type, label))
    return list(found)


# ---------------------------------------------------------------------------
# Graph retrieval (SPARQL)
# ---------------------------------------------------------------------------

SPARQL_CANDIDATE_URLS = """
PREFIX horatio: <https://horatio.ai/ontology/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?url WHERE {{
    ?concept rdfs:label ?clabel .
    FILTER({filter_clause})
    ?section horatio:mentions ?concept ;
             horatio:sourceUrl ?url .
}}
"""

SPARQL_CONCEPT_CONTEXT = """
PREFIX horatio: <https://horatio.ai/ontology/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?type ?clabel WHERE {{
    ?page horatio:sourceUrl ?url .
    FILTER(?url IN ({url_list}))
    ?page horatio:hasSection ?section .
    ?section horatio:mentions ?concept .
    ?concept a ?type ; rdfs:label ?clabel .
    FILTER(STRSTARTS(STR(?type), "https://horatio.ai/ontology/"))
}}
ORDER BY ?type ?clabel
"""


def graph_candidate_urls(
    g: Graph,
    concepts: list[tuple[str, str]],
) -> list[str] | None:
    """Return candidate page URLs from the graph for the detected concepts.

    Returns None if no concepts were detected (triggers unscoped search).
    Returns an empty list if concepts were detected but nothing matched
    (triggers unscoped fallback to avoid returning zero results).
    """
    if not concepts:
        return None

    labels = [label for _, label in concepts]
    filter_clause = " || ".join(f'?clabel = "{label}"' for label in labels)
    q = SPARQL_CANDIDATE_URLS.format(filter_clause=filter_clause)
    urls = [str(row.url) for row in g.query(q)]
    return urls


def graph_concept_context(
    g: Graph,
    urls: list[str],
) -> list[tuple[str, str]]:
    """Return all concepts mentioned on the candidate pages.

    Used to enrich the synthesis prompt with structured context about
    what the retrieved pages cover, beyond what the chunks themselves say.
    """
    if not urls:
        return []
    url_list = ", ".join(f'<{u}>' for u in urls)
    q = SPARQL_CONCEPT_CONTEXT.format(url_list=url_list)
    return [
        (str(row.type).split("/")[-1], str(row.clabel))
        for row in g.query(q)
    ]


# ---------------------------------------------------------------------------
# Vector retrieval (ChromaDB)
# ---------------------------------------------------------------------------

def vector_retrieve(
    collection: chromadb.Collection,
    question: str,
    top_k: int,
    candidate_urls: list[str] | None,
) -> list[dict]:
    """Query ChromaDB and return the top-k chunks.

    If candidate_urls is provided and non-empty, restricts results to those
    pages via a metadata where filter.  Falls back to unscoped search if
    candidate_urls is empty (graph matched concepts but found no pages).
    """
    where = None
    scoped = False

    if candidate_urls:
        scoped = True
        if len(candidate_urls) == 1:
            where = {"source_url": {"$eq": candidate_urls[0]}}
        else:
            where = {"source_url": {"$in": candidate_urls}}

    kwargs: dict = dict(
        query_texts=[question],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "body":          doc,
            "distance":      dist,
            "source_url":    meta.get("source_url", ""),
            "source_title":  meta.get("source_title", ""),
            "heading_text":  meta.get("heading_text", ""),
            "heading_path":  meta.get("heading_path", ""),
            "content_type":  meta.get("content_type", ""),
            "chunk_index":   meta.get("chunk_index", 0),
            "chunk_total":   meta.get("chunk_total", 1),
            "scoped":        scoped,
        })
    return chunks


# ---------------------------------------------------------------------------
# LLM synthesis
# ---------------------------------------------------------------------------

def format_chunks_for_prompt(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        path = chunk["heading_path"]
        heading = chunk["heading_text"]
        full_path = f"{path} | {heading}" if path else heading
        chunk_note = (
            f" (part {chunk['chunk_index'] + 1} of {chunk['chunk_total']})"
            if chunk["chunk_total"] > 1 else ""
        )
        parts.append(
            f"--- Chunk {i}{chunk_note} ---\n"
            f"Page: {chunk['source_title']}\n"
            f"Section: {full_path}\n"
            f"URL: {chunk['source_url']}\n"
            f"Content:\n{chunk['body']}"
        )
    return "\n\n".join(parts)


def synthesize(
    client: anthropic.Anthropic,
    question: str,
    chunks: list[dict],
    concept_context: list[tuple[str, str]],
    model: str,
    max_tokens: int,
) -> str:
    """Send retrieved chunks to Claude and return a cited answer."""
    context_note = ""
    if concept_context:
        by_type: dict[str, list[str]] = {}
        for ctype, clabel in concept_context:
            by_type.setdefault(ctype, []).append(clabel)
        lines = [
            f"  {ctype}s: {', '.join(sorted(labels))}"
            for ctype, labels in sorted(by_type.items())
        ]
        context_note = (
            "\nGraph context (concepts mentioned on retrieved pages):\n"
            + "\n".join(lines) + "\n"
        )

    formatted_chunks = format_chunks_for_prompt(chunks)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        question=question,
        chunks=context_note + formatted_chunks,
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

def display_retrieval(
    question: str,
    concepts: list[tuple[str, str]],
    candidate_urls: list[str] | None,
    chunks: list[dict],
) -> None:
    print(f"\n{'─' * 60}")

    if concepts:
        concept_str = ", ".join(f"{t}:{l}" for t, l in sorted(concepts))
        print(f"Concepts detected:  {concept_str}")
        if candidate_urls is not None:
            scope = "scoped" if candidate_urls else "unscoped (fallback)"
            print(f"Graph candidate pages: {len(candidate_urls)}  [{scope}]")
    else:
        print("Concepts detected:  none  [unscoped vector search]")

    scoped = any(c["scoped"] for c in chunks)
    print(f"Retrieved {len(chunks)} chunks ({'scoped' if scoped else 'unscoped'}):")

    for i, chunk in enumerate(chunks, start=1):
        path = chunk["heading_path"]
        heading = chunk["heading_text"]
        full_path = f"{path} | {heading}" if path else heading
        print(f"\n  {i}. [{chunk['distance']:.4f}] {full_path}")
        print(f"     {chunk['source_title']} — {chunk['source_url']}")
        print(f"     {chunk['body'][:100]}...")


def display_answer(answer: str) -> None:
    print(f"\n{'─' * 60}")
    print("Answer:\n")
    print(answer)
    print(f"{'─' * 60}")


# ---------------------------------------------------------------------------
# Query orchestration
# ---------------------------------------------------------------------------

def run_query(
    question: str,
    g: Graph | None,
    collection: chromadb.Collection,
    anthropic_client: anthropic.Anthropic | None,
    top_k: int,
    model: str,
    max_tokens: int,
    show_retrieval: bool,
    no_synthesis: bool,
    no_graph: bool,
) -> None:
    # --- Stage 1: concept detection + graph pre-filtering ---
    concepts: list[tuple[str, str]] = []
    candidate_urls: list[str] | None = None
    concept_context: list[tuple[str, str]] = []

    if g is not None and not no_graph:
        concepts = detect_concepts(question)
        candidate_urls = graph_candidate_urls(g, concepts)

        # If graph returned zero URLs for detected concepts, fall back to
        # unscoped search rather than returning nothing.
        if candidate_urls is not None and len(candidate_urls) == 0:
            candidate_urls = None

        if candidate_urls:
            concept_context = graph_concept_context(g, candidate_urls)

    # --- Stage 2: scoped vector search ---
    chunks = vector_retrieve(collection, question, top_k, candidate_urls)

    # --- Display retrieval info ---
    if show_retrieval or no_synthesis:
        display_retrieval(question, concepts, candidate_urls, chunks)

    if no_synthesis or anthropic_client is None:
        return

    # --- Stage 3: LLM synthesis ---
    answer = synthesize(
        anthropic_client, question, chunks,
        concept_context, model, max_tokens,
    )
    if not (show_retrieval or no_synthesis):
        # Print a compact retrieval summary before the answer
        scope_info = (
            f"graph-scoped to {len(candidate_urls)} pages"
            if candidate_urls else "unscoped vector search"
        )
        concept_str = (
            ", ".join(l for _, l in sorted(concepts))
            if concepts else "none"
        )
        print(f"\n[concepts: {concept_str} | {scope_info} | {len(chunks)} chunks]")

    display_answer(answer)


def run_repl(
    g: Graph | None,
    collection: chromadb.Collection,
    anthropic_client: anthropic.Anthropic | None,
    top_k: int,
    model: str,
    max_tokens: int,
    show_retrieval: bool,
    no_synthesis: bool,
    no_graph: bool,
) -> None:
    graph_status = "enabled" if (g is not None and not no_graph) else "disabled"
    print("\nHoratio hybrid query interface")
    print(f"Collection: {collection.name}  |  Top-k: {top_k}  |  "
          f"Model: {model}  |  Graph: {graph_status}")
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
            g=g,
            collection=collection,
            anthropic_client=anthropic_client,
            top_k=top_k,
            model=model,
            max_tokens=max_tokens,
            show_retrieval=show_retrieval,
            no_synthesis=no_synthesis,
            no_graph=no_graph,
        )
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hybrid query interface for the Horatio neuro-symbolic pipeline. "
            "Combines SPARQL graph pre-filtering with ChromaDB vector search "
            "and Claude answer synthesis."
        )
    )
    parser.add_argument(
        "--query", "-q",
        type=str, default=None,
        help="Run a single query and exit (omit for interactive REPL).",
    )
    parser.add_argument(
        "--db-dir",
        type=Path, default=DEFAULT_DB_DIR,
        help=f"ChromaDB directory (default: {DEFAULT_DB_DIR})",
    )
    parser.add_argument(
        "--graph",
        type=Path, default=DEFAULT_GRAPH_FILE,
        help=f"RDF graph Turtle file (default: {DEFAULT_GRAPH_FILE})",
    )
    parser.add_argument(
        "--collection",
        type=str, default=DEFAULT_COLLECTION,
        help=f"ChromaDB collection name (default: {DEFAULT_COLLECTION})",
    )
    parser.add_argument(
        "--embedding-model",
        type=str, default=DEFAULT_EMBEDDING_MODEL,
        help=f"Sentence-transformers model (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--model",
        type=str, default=DEFAULT_LLM_MODEL,
        help=f"Claude model for synthesis (default: {DEFAULT_LLM_MODEL})",
    )
    parser.add_argument(
        "--top-k",
        type=int, default=DEFAULT_TOP_K,
        help=f"Chunks to retrieve (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int, default=DEFAULT_MAX_TOKENS,
        help=f"Max tokens in Claude response (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--show-retrieval",
        action="store_true",
        help="Print retrieved chunks and graph scoping info before the answer.",
    )
    parser.add_argument(
        "--no-synthesis",
        action="store_true",
        help="Skip LLM synthesis; show only retrieved chunks.",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Disable graph pre-filtering; use pure vector search (baseline mode).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    # --- Anthropic client ---
    anthropic_client = None
    if not args.no_synthesis:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print(
                "Error: ANTHROPIC_API_KEY not set.\n"
                "Set it with: set -Ux ANTHROPIC_API_KEY sk-ant-...\n"
                "Or run with --no-synthesis to skip the LLM step.",
                file=sys.stderr,
            )
            return 2
        anthropic_client = anthropic.Anthropic(api_key=api_key)

    # --- RDF graph ---
    g: Graph | None = None
    if not args.no_graph:
        if not args.graph.exists():
            print(
                f"Warning: graph file not found: {args.graph}\n"
                "Falling back to vector-only search.\n"
                "Run build-graph.py to create the graph.",
                file=sys.stderr,
            )
        else:
            print(f"Loading graph: {args.graph}", flush=True)
            g = Graph()
            g.parse(str(args.graph), format="turtle")
            print(f"Graph loaded: {len(g)} triples.")

    # --- ChromaDB ---
    if not args.db_dir.exists():
        print(
            f"Error: ChromaDB directory not found: {args.db_dir}\n"
            "Run embed-chunks.py first.",
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
            f"Error: Collection '{args.collection}' not found.\n"
            "Run embed-chunks.py first.",
            file=sys.stderr,
        )
        return 2

    print(f"Collection '{args.collection}': {collection.count()} documents.\n")

    # --- Single query or REPL ---
    if args.query:
        run_query(
            question=args.query,
            g=g,
            collection=collection,
            anthropic_client=anthropic_client,
            top_k=args.top_k,
            model=args.model,
            max_tokens=args.max_tokens,
            show_retrieval=args.show_retrieval,
            no_synthesis=args.no_synthesis,
            no_graph=args.no_graph,
        )
    else:
        run_repl(
            g=g,
            collection=collection,
            anthropic_client=anthropic_client,
            top_k=args.top_k,
            model=args.model,
            max_tokens=args.max_tokens,
            show_retrieval=args.show_retrieval,
            no_synthesis=args.no_synthesis,
            no_graph=args.no_graph,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())