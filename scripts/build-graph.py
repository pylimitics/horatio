#!/usr/bin/env python3
"""Build an RDF knowledge graph from the Horatio chunks JSON.

Populates a persistent RDF graph (serialised as Turtle) with two phases
of content:

  Phase 1 — Structural graph (fully automatic)
  ─────────────────────────────────────────────
  Nodes derived directly from the chunks JSON with no inference:
    :Book       — the top-level documentation set (one per root URL)
    :Page       — one DITA topic page, identified by URL
    :Section    — a headed section within a page

  Relationships:
    (:Book)-[:contains]->(:Page)
    (:Page)-[:hasSection]->(:Section)
    (:Section)-[:hasSubsection]->(:Section)   derived from heading_path
    (:Page)-[:precedesPage]->(:Page)          TOC order

  Phase 2 — Concept graph (rule-based extraction)
  ─────────────────────────────────────────────────
  Named concepts extracted from section bodies using known term lists:
    :Product    — named Cisco security products
    :Role       — named user roles and administrator types
    :Feature    — named platform features

  Relationships:
    (:Section)-[:mentions]->(:Concept)

  Concept-to-concept relationships (prerequisites, dependencies) are
  deferred to a future LLM-assisted extraction phase.

Usage:
    pip3 install rdflib
    python3 build-graph.py data/horatio/chunks-2026-06-05.json
    python3 build-graph.py data/horatio/chunks-2026-06-05.json --corpus data/horatio/corpus-2026-06-05.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef, XSD
from rdflib.namespace import DC

# ---------------------------------------------------------------------------
# Namespaces
# ---------------------------------------------------------------------------

HORATIO = Namespace("https://horatio.ai/ontology/")
SCC     = Namespace("https://securitydocs.cisco.com/docs/scc/")
DATA    = Namespace("https://horatio.ai/data/")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path("data/horatio")
DEFAULT_GRAPH_FILE = "graph.ttl"

# ---------------------------------------------------------------------------
# Known concepts
# Enumerated from corpus inspection.  Extend as additional terms are found.
# Each entry is (canonical_label, [aliases_to_match_in_text])
# ---------------------------------------------------------------------------

KNOWN_PRODUCTS: list[tuple[str, list[str]]] = [
    ("AI Defense",                      ["AI Defense"]),
    ("Firewall Management",             ["Firewall Management", "FMC"]),
    ("Multicloud Defense",              ["Multicloud Defense"]),
    ("Secure Access",                   ["Secure Access"]),
    ("Secure Workload",                 ["Secure Workload"]),
    ("Secure Email Threat Defense",     ["Secure Email Threat Defense"]),
    ("Secure Endpoint",                 ["Secure Endpoint"]),
    ("Cisco Duo",                       ["Cisco Duo", "Duo"]),
    ("Cisco XDR",                       ["Cisco XDR", "XDR"]),
    ("SASE Management",                 ["SASE Management", "SASE", "Secure Access Service Edge"]),
    ("SD-WAN",                          ["SD-WAN", "Catalyst SD-WAN"]),
]

KNOWN_ROLES: list[tuple[str, list[str]]] = [
    ("Organization Administrator",      ["organization administrator", "org admin"]),
    ("Super Admin",                     ["super admin", "SuperAdmin"]),
    ("Read Only User",                  ["read only", "read-only user"]),
]

KNOWN_FEATURES: list[tuple[str, list[str]]] = [
    ("Role-Based Access Control",       ["role-based access control", "RBAC"]),
    ("Platform Management",             ["Platform Management"]),
    ("AI Assistant",                    ["AI Assistant", "Cisco AI Assistant"]),
    ("Global Search",                   ["Global Search"]),
    ("Shared Objects",                  ["Shared Objects"]),
    ("Multi-Org",                       ["Multi-Org", "multi-org", "multi-organization"]),
    ("Claim Code",                      ["claim code"]),
    ("Identity Provider",               ["identity provider", "IdP", "SAML"]),
    ("Single Sign-On",                  ["single sign-on", "SSO", "Security Cloud Sign On"]),
]

# ---------------------------------------------------------------------------
# URI construction helpers
# ---------------------------------------------------------------------------

def page_uri(url: str) -> URIRef:
    """Stable URI for a Page node derived from its source URL."""
    slug = hashlib.sha256(url.encode()).hexdigest()[:16]
    return DATA[f"page/{slug}"]


def section_uri(source_url: str, heading_path: list[str], heading_text: str) -> URIRef:
    """Stable URI for a Section node."""
    key = f"{source_url}::{'|'.join(heading_path)}::{heading_text}"
    slug = hashlib.sha256(key.encode()).hexdigest()[:16]
    return DATA[f"section/{slug}"]


def concept_uri(concept_type: str, label: str) -> URIRef:
    """Stable URI for a Concept node."""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", label).strip("-")
    return DATA[f"{concept_type.lower()}/{slug}"]


def book_uri(root_url: str) -> URIRef:
    slug = hashlib.sha256(root_url.encode()).hexdigest()[:16]
    return DATA[f"book/{slug}"]


# ---------------------------------------------------------------------------
# Concept matching
# ---------------------------------------------------------------------------

def build_concept_index(
    products: list[tuple[str, list[str]]],
    roles: list[tuple[str, list[str]]],
    features: list[tuple[str, list[str]]],
) -> list[tuple[re.Pattern, str, str]]:
    """Build a list of (pattern, concept_type, canonical_label) tuples."""
    index = []
    for label, aliases in products:
        for alias in aliases:
            pattern = re.compile(r'\b' + re.escape(alias) + r'\b')
            index.append((pattern, "Product", label))
    for label, aliases in roles:
        for alias in aliases:
            pattern = re.compile(r'\b' + re.escape(alias) + r'\b', re.IGNORECASE)
            index.append((pattern, "Role", label))
    for label, aliases in features:
        for alias in aliases:
            pattern = re.compile(r'\b' + re.escape(alias) + r'\b', re.IGNORECASE)
            index.append((pattern, "Feature", label))
    return index


def find_concepts(
    text: str,
    concept_index: list[tuple[re.Pattern, str, str]],
) -> list[tuple[str, str]]:
    """Return list of (concept_type, canonical_label) found in text."""
    found = set()
    for pattern, concept_type, label in concept_index:
        if pattern.search(text):
            found.add((concept_type, label))
    return list(found)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(
    chunks_data: dict,
    corpus_data: dict | None,
) -> tuple[Graph, dict]:
    g = Graph()
    g.bind("horatio", HORATIO)
    g.bind("scc", SCC)
    g.bind("data", DATA)
    g.bind("dc", DC)

    concept_index = build_concept_index(KNOWN_PRODUCTS, KNOWN_ROLES, KNOWN_FEATURES)

    stats = {
        "books": 0,
        "pages": 0,
        "sections": 0,
        "concept_mentions": 0,
        "concepts": {"Product": set(), "Role": set(), "Feature": set()},
    }

    # --- Define ontology classes ---
    for cls in ["Book", "Page", "Section", "Product", "Role", "Feature"]:
        g.add((HORATIO[cls], RDF.type, RDFS.Class))
        g.add((HORATIO[cls], RDFS.label, Literal(cls)))

    # --- Define ontology properties ---
    properties = [
        ("contains",     "Book contains Page"),
        ("hasSection",   "Page has Section"),
        ("hasSubsection","Section has Subsection"),
        ("precedesPage", "Page precedes Page in TOC order"),
        ("mentions",     "Section mentions Concept"),
        ("sourceUrl",    "Source URL of a node"),
        ("headingText",  "Heading text of a Section"),
        ("headingPath",  "Ancestor heading path of a Section"),
        ("headingLevel", "Heading level of a Section"),
        ("tocOrder",     "Position of Page in TOC"),
        ("contentType",  "Content type of a Section chunk"),
    ]
    for prop_name, comment in properties:
        g.add((HORATIO[prop_name], RDF.type, RDF.Property))
        g.add((HORATIO[prop_name], RDFS.comment, Literal(comment)))

    # --- Book node ---
    root_url = chunks_data["root_url"]
    book = book_uri(root_url)
    g.add((book, RDF.type, HORATIO["Book"]))
    g.add((book, RDFS.label, Literal("Security Cloud Control Getting Started Guide")))
    g.add((book, HORATIO["sourceUrl"], Literal(root_url, datatype=XSD.anyURI)))
    g.add((book, DC.date, Literal(chunks_data.get("processed_at", ""), datatype=XSD.string)))
    stats["books"] += 1

    # --- Build page order from corpus TOC if available ---
    # The corpus pages list preserves TOC order from capture-content.py.
    # When corpus is not provided, derive order from chunk appearance order.
    toc_order: dict[str, int] = {}
    if corpus_data and "pages" in corpus_data:
        for i, p in enumerate(corpus_data["pages"]):
            toc_order[p["url"]] = i
    else:
        # Fallback: assign order by first appearance in chunks list
        order_counter = 0
        for chunk in chunks_data.get("chunks", []):
            url = chunk["source_url"]
            if url not in toc_order:
                toc_order[url] = order_counter
                order_counter += 1

    # --- Collect pages and sections from chunks ---
    # Group chunks by source_url to reconstruct page structure
    pages_seen: dict[str, URIRef] = {}
    page_titles: dict[str, str] = {}
    # section URI -> list of (heading_path, heading_text) for subsection edges
    section_paths: dict[str, tuple[list[str], str]] = {}

    # Accept either chunks JSON (from process-corpus.py, has "chunks" key)
    # or corpus JSON (from capture-content.py, has "pages" key).
    if "chunks" in chunks_data:
        chunks = chunks_data["chunks"]
    elif "pages" in chunks_data:
        # Reconstruct chunk-like dicts from corpus sections directly
        chunks = []
        for page in chunks_data["pages"]:
            for section in page["sections"]:
                chunks.append({
                    "source_url":    section["source_url"],
                    "source_title":  section["source_title"],
                    "heading_text":  section["heading_text"],
                    "heading_path":  section["heading_path"],
                    "heading_level": section["heading_level"],
                    "body":          section["body"],
                    "chunk_index":   0,
                    "chunk_total":   1,
                    "content_type":  "unsplit",
                })
    else:
        print("Error: input JSON has neither 'chunks' nor 'pages' key.", file=sys.stderr)
        return 2

    for chunk in chunks:
        source_url  = chunk["source_url"]
        source_title = chunk["source_title"]
        heading_text = chunk["heading_text"]
        heading_path = chunk["heading_path"]
        heading_level = chunk["heading_level"]
        body         = chunk["body"]
        content_type = chunk["content_type"]

        # --- Page node ---
        if source_url not in pages_seen:
            p_uri = page_uri(source_url)
            pages_seen[source_url] = p_uri
            page_titles[source_url] = source_title
            g.add((p_uri, RDF.type, HORATIO["Page"]))
            g.add((p_uri, RDFS.label, Literal(source_title)))
            g.add((p_uri, HORATIO["sourceUrl"], Literal(source_url, datatype=XSD.anyURI)))
            order = toc_order.get(source_url, -1)
            if order >= 0:
                g.add((p_uri, HORATIO["tocOrder"], Literal(order, datatype=XSD.integer)))
            g.add((book, HORATIO["contains"], p_uri))
            stats["pages"] += 1

        p_uri = pages_seen[source_url]

        # --- Section node ---
        s_uri = section_uri(source_url, heading_path, heading_text)
        # Sections may span multiple chunks (chunk_total > 1); only add the
        # node once (on chunk_index 0) but always add concept mentions.
        if chunk["chunk_index"] == 0:
            g.add((s_uri, RDF.type, HORATIO["Section"]))
            g.add((s_uri, RDFS.label, Literal(heading_text)))
            g.add((s_uri, HORATIO["headingText"], Literal(heading_text)))
            g.add((s_uri, HORATIO["headingPath"], Literal(" | ".join(heading_path))))
            g.add((s_uri, HORATIO["headingLevel"], Literal(heading_level, datatype=XSD.integer)))
            g.add((s_uri, HORATIO["contentType"], Literal(content_type)))
            g.add((s_uri, HORATIO["sourceUrl"], Literal(source_url, datatype=XSD.anyURI)))
            g.add((p_uri, HORATIO["hasSection"], s_uri))
            section_paths[str(s_uri)] = (heading_path, heading_text)
            stats["sections"] += 1

        # --- Concept mentions (all chunks, to catch mentions in split bodies) ---
        concepts = find_concepts(body, concept_index)
        for concept_type, concept_label in concepts:
            c_uri = concept_uri(concept_type, concept_label)
            if (c_uri, RDF.type, HORATIO[concept_type]) not in g:
                g.add((c_uri, RDF.type, HORATIO[concept_type]))
                g.add((c_uri, RDFS.label, Literal(concept_label)))
                stats["concepts"][concept_type].add(concept_label)
            g.add((s_uri, HORATIO["mentions"], c_uri))
            stats["concept_mentions"] += 1

    # --- Subsection edges ---
    # For each section with a non-empty heading_path, find the parent section
    # (the section whose heading_text == last element of this section's path)
    # and add a hasSubsection edge.
    for s_uri_str, (heading_path, heading_text) in section_paths.items():
        if not heading_path:
            continue
        parent_heading_text = heading_path[-1]
        parent_path = heading_path[:-1]
        # Find the parent section URI by reconstructing it
        # We need the source_url — get it from the section's sourceUrl triple
        s_uri = URIRef(s_uri_str)
        source_url_literal = g.value(s_uri, HORATIO["sourceUrl"])
        if source_url_literal is None:
            continue
        source_url = str(source_url_literal)
        parent_uri = section_uri(source_url, parent_path, parent_heading_text)
        if (parent_uri, RDF.type, HORATIO["Section"]) in g:
            g.add((parent_uri, HORATIO["hasSubsection"], s_uri))

    # --- TOC order edges (precedesPage) ---
    # Sort pages by tocOrder and add sequential edges
    ordered_pages = sorted(
        [(toc_order.get(url, 9999), uri) for url, uri in pages_seen.items()]
    )
    for i in range(len(ordered_pages) - 1):
        _, current = ordered_pages[i]
        _, next_page = ordered_pages[i + 1]
        g.add((current, HORATIO["precedesPage"], next_page))

    # Finalise concept stats
    for concept_type in stats["concepts"]:
        stats["concepts"][concept_type] = len(stats["concepts"][concept_type])

    return g, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an RDF knowledge graph from the Horatio chunks JSON. "
            "Produces a Turtle (.ttl) file containing structural and concept "
            "graph data for the Horatio neuro-symbolic pipeline."
        )
    )
    parser.add_argument(
        "chunks",
        type=Path,
        help="Path to the chunks JSON produced by process-corpus.py",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help=(
            "Path to the corpus JSON produced by capture-content.py. "
            "Used to preserve TOC page ordering for precedesPage edges. "
            "Optional but recommended."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            f"Output path for the Turtle graph file. "
            f"Default: <chunks-dir>/{DEFAULT_GRAPH_FILE}"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.chunks.exists():
        print(f"Error: chunks file not found: {args.chunks}", file=sys.stderr)
        return 2

    print(f"Reading chunks: {args.chunks}", flush=True)
    chunks_data = json.loads(args.chunks.read_text(encoding="utf-8"))

    corpus_data = None
    if args.corpus:
        if not args.corpus.exists():
            print(f"Error: corpus file not found: {args.corpus}", file=sys.stderr)
            return 2
        print(f"Reading corpus: {args.corpus}", flush=True)
        corpus_data = json.loads(args.corpus.read_text(encoding="utf-8"))

    print("Building RDF graph...", flush=True)
    g, stats = build_graph(chunks_data, corpus_data)

    output = args.output or (args.chunks.parent / DEFAULT_GRAPH_FILE)
    output.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(output), format="turtle")

    triple_count = len(g)
    print(f"\nGraph construction complete:")
    print(f"  Books:            {stats['books']}")
    print(f"  Pages:            {stats['pages']}")
    print(f"  Sections:         {stats['sections']}")
    print(f"  Concept mentions: {stats['concept_mentions']}")
    print(f"  Concepts found:")
    for concept_type, count in stats["concepts"].items():
        print(f"    {concept_type}: {count}")
    print(f"  Total triples:    {triple_count}")
    print(f"\nWrote graph: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())