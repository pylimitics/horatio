#!/usr/bin/env python3
"""Process a raw Horatio JSON corpus into pipeline-ready chunks.

Reads the JSON produced by capture-content.py and applies two transforms:

  1. Filter — drop sections whose body is empty or below a minimum length.
     These are DITA container topics (TOC nodes with no body content) that
     carry no retrievable information.

  2. Split — divide sections whose body exceeds a maximum character length
     into sub-chunks that fit within an embedding model's context window.
     Three splitting strategies are applied based on detected content type:

       prose      — split on paragraph (newline) boundaries, grouping lines
                    into chunks that stay below the character ceiling.

       procedure  — reassemble numbered steps (solo-digit lines followed by
                    their content lines) before splitting, so step numbers
                    are never separated from their text.

       code/XML   — treat the entire body as one atomic chunk regardless of
                    length, since splitting mid-tag or mid-block would corrupt
                    the content.  A warning is emitted if the block exceeds
                    the ceiling.

Each output chunk carries the full citation context needed by the retrieval
pipeline: source URL, source title, heading path, heading text, and a
chunk_index (0-based) for sections that were split into more than one chunk.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Sections with bodies shorter than this are dropped entirely.
# Covers empty container topics and any near-empty stubs.
DEFAULT_MIN_BODY_CHARS = 50

# Sections with bodies longer than this are split into sub-chunks.
# 800 characters is approximately 200 tokens, comfortably inside the
# 256-token sweet spot for all-MiniLM-L6-v2 and well inside the 512-token
# limit for multi-qa-mpnet-base-dot-v1.
DEFAULT_MAX_CHUNK_CHARS = 800

# A line containing only digits (a DITA ordered-list step number) with
# fewer digits than this is treated as a step-number marker, not content.
# Prevents false positives on lines that happen to be short numbers in prose.
MAX_STEP_NUMBER = 99


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Chunk:
    """A single pipeline-ready unit of content.

    One Section produces one or more Chunks.  Sections that are not split
    produce exactly one Chunk with chunk_index 0 and chunk_total 1.
    """
    # Citation fields — carried from the source Section unchanged.
    source_url: str
    source_title: str
    heading_path: list[str]
    heading_text: str
    heading_level: int

    # The text content for this chunk.
    body: str

    # Position within the parent section's split sequence.
    # chunk_index is 0-based; chunk_total is the total number of chunks
    # the parent section was split into.
    chunk_index: int
    chunk_total: int

    # Content type inferred during splitting.  Recorded for inspection and
    # for downstream decisions (e.g. whether to apply special graph modelling
    # to procedure steps).
    content_type: str   # "prose" | "procedure" | "code" | "unsplit"


# ---------------------------------------------------------------------------
# Content type detection
# ---------------------------------------------------------------------------

def detect_content_type(body: str) -> str:
    """Classify a section body as 'code', 'procedure', or 'prose'."""
    if _looks_like_code(body):
        return "code"
    if _looks_like_procedure(body):
        return "procedure"
    return "prose"


def _looks_like_code(body: str) -> bool:
    """True if the body appears to contain a code or XML block."""
    return (
        "<?xml" in body
        or ("</" in body and "/>" in body)
        or body.lstrip().startswith("<")
    )


def _looks_like_procedure(body: str) -> bool:
    """True if the body appears to be a numbered procedure.

    Requires at least three solo-digit step-number lines so that a body
    that happens to contain one or two standalone numbers doesn't get
    misclassified.
    """
    lines = body.split("\n")
    solo_digit_count = sum(
        1 for line in lines
        if line.strip().isdigit() and int(line.strip()) <= MAX_STEP_NUMBER
    )
    return solo_digit_count >= 3


# ---------------------------------------------------------------------------
# Splitting strategies
# ---------------------------------------------------------------------------

def split_prose(body: str, max_chars: int) -> list[str]:
    """Split prose into chunks on paragraph (line) boundaries.

    Groups consecutive lines into a chunk until adding the next line would
    exceed max_chars.  A single line that exceeds max_chars is kept as its
    own chunk rather than being broken mid-sentence.
    """
    lines = [line for line in body.split("\n") if line.strip()]
    if not lines:
        return []

    chunks: list[str] = []
    current_lines: list[str] = []
    current_len = 0

    for line in lines:
        # +1 for the newline that will join the lines.
        addition = len(line) + (1 if current_lines else 0)
        if current_lines and current_len + addition > max_chars:
            chunks.append("\n".join(current_lines))
            current_lines = [line]
            current_len = len(line)
        else:
            current_lines.append(line)
            current_len += addition

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks


def split_procedure(body: str, max_chars: int) -> list[str]:
    """Split a numbered procedure into chunks, keeping steps intact.

    Pass 1 — reassemble logical steps: each solo-digit step-number line is
    joined with all following content lines until the next step number.
    Consecutive step numbers with no content between them (DITA steps whose
    body is entirely in a sub-element) are merged into one unit.

    Pass 2 — expand oversized steps: any single step whose text exceeds
    max_chars is prose-split, with the step number preserved as a prefix on
    the first sub-chunk so the citation context is not lost.

    Pass 3 — group into chunks: expanded steps are accumulated into chunks
    using the same boundary logic as split_prose.
    """
    lines = body.split("\n")

    # --- Pass 1: reassemble logical steps ---
    steps: list[str] = []
    current_step_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        is_step_number = stripped.isdigit() and int(stripped) <= MAX_STEP_NUMBER

        if is_step_number:
            if current_step_lines and not all(
                l.strip().isdigit() for l in current_step_lines if l.strip()
            ):
                steps.append("\n".join(
                    l for l in current_step_lines if l.strip()
                ))
                current_step_lines = []
            current_step_lines.append(stripped)
        else:
            if stripped:
                current_step_lines.append(stripped)

    if current_step_lines:
        content = "\n".join(l for l in current_step_lines if l.strip())
        if content:
            steps.append(content)

    if not steps:
        return split_prose(body, max_chars)

    # --- Pass 2: expand any step that individually exceeds max_chars ---
    expanded_steps: list[str] = []
    for step in steps:
        if len(step) <= max_chars:
            expanded_steps.append(step)
            continue
        step_lines = [l for l in step.split("\n") if l.strip()]
        if step_lines and step_lines[0].isdigit():
            # Keep the step number attached to the first prose sub-chunk.
            prefix = step_lines[0]
            rest = "\n".join(step_lines[1:])
            sub_chunks = split_prose(rest, max_chars - len(prefix) - 1)
            if sub_chunks:
                expanded_steps.append(f"{prefix}\n{sub_chunks[0]}")
                expanded_steps.extend(sub_chunks[1:])
            else:
                expanded_steps.append(step)
        else:
            expanded_steps.extend(split_prose(step, max_chars))

    # --- Pass 3: group expanded steps into chunks ---
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for step in expanded_steps:
        addition = len(step) + (2 if current_parts else 0)  # 2 for "\n\n"
        if current_parts and current_len + addition > max_chars:
            chunks.append("\n\n".join(current_parts))
            current_parts = [step]
            current_len = len(step)
        else:
            current_parts.append(step)
            current_len += addition

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


def split_code(body: str, max_chars: int) -> list[str]:
    """Return the code body as a single atomic chunk.

    Code and XML blocks must not be split because doing so would produce
    fragments that are syntactically invalid and unembeddable as meaningful
    units.  The block is returned whole regardless of length.
    """
    return [body.strip()]


# ---------------------------------------------------------------------------
# Section processing
# ---------------------------------------------------------------------------

def process_section(section: dict[str, Any], min_chars: int, max_chars: int) -> list[Chunk]:
    """Convert one corpus section into zero or more pipeline chunks.

    Returns an empty list if the section is filtered out.
    Returns a list of one Chunk if no splitting is needed.
    Returns a list of multiple Chunks if the body was split.
    """
    body = section["body"].strip()

    # --- Filter ---
    if len(body) < min_chars:
        return []

    # --- Detect content type ---
    content_type = detect_content_type(body)

    # --- Split if needed ---
    if len(body) <= max_chars:
        return [_make_chunk(section, body, 0, 1, "unsplit")]

    if content_type == "code":
        raw_chunks = split_code(body, max_chars)
    elif content_type == "procedure":
        raw_chunks = split_procedure(body, max_chars)
    else:
        raw_chunks = split_prose(body, max_chars)

    # Filter any empty strings that splitting may produce.
    raw_chunks = [c for c in raw_chunks if c.strip()]

    if not raw_chunks:
        return []

    return [
        _make_chunk(section, chunk_body, idx, len(raw_chunks), content_type)
        for idx, chunk_body in enumerate(raw_chunks)
    ]


def _make_chunk(
    section: dict[str, Any],
    body: str,
    chunk_index: int,
    chunk_total: int,
    content_type: str,
) -> Chunk:
    return Chunk(
        source_url=section["source_url"],
        source_title=section["source_title"],
        heading_path=section["heading_path"],
        heading_text=section["heading_text"],
        heading_level=section["heading_level"],
        body=body,
        chunk_index=chunk_index,
        chunk_total=chunk_total,
        content_type=content_type,
    )


# ---------------------------------------------------------------------------
# Corpus processing
# ---------------------------------------------------------------------------

def process_corpus(
    corpus: dict[str, Any],
    min_chars: int,
    max_chars: int,
) -> tuple[list[Chunk], dict[str, Any]]:
    """Process all sections in a corpus and return chunks plus a stats dict."""
    all_chunks: list[Chunk] = []

    stats: dict[str, Any] = {
        "pages": len(corpus["pages"]),
        "sections_total": 0,
        "sections_filtered": 0,
        "sections_unsplit": 0,
        "sections_split": 0,
        "chunks_total": 0,
        "content_types": {"prose": 0, "procedure": 0, "code": 0, "unsplit": 0},
        "code_blocks_over_limit": [],
    }

    for page in corpus["pages"]:
        for section in page["sections"]:
            stats["sections_total"] += 1
            body = section["body"].strip()

            if len(body) < min_chars:
                stats["sections_filtered"] += 1
                continue

            chunks = process_section(section, min_chars, max_chars)
            if not chunks:
                stats["sections_filtered"] += 1
                continue

            if len(chunks) == 1:
                stats["sections_unsplit"] += 1
            else:
                stats["sections_split"] += 1

            for chunk in chunks:
                stats["content_types"][chunk.content_type] += 1
                # Warn about code blocks that exceed the limit — these cannot
                # be split, so the embedding model will truncate them.
                if chunk.content_type == "code" and len(chunk.body) > max_chars:
                    stats["code_blocks_over_limit"].append({
                        "source_url": chunk.source_url,
                        "heading_text": chunk.heading_text,
                        "body_chars": len(chunk.body),
                    })

            all_chunks.extend(chunks)

    stats["chunks_total"] = len(all_chunks)
    return all_chunks, stats


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    return {
        "source_url": chunk.source_url,
        "source_title": chunk.source_title,
        "heading_path": chunk.heading_path,
        "heading_text": chunk.heading_text,
        "heading_level": chunk.heading_level,
        "body": chunk.body,
        "chunk_index": chunk.chunk_index,
        "chunk_total": chunk.chunk_total,
        "content_type": chunk.content_type,
    }


def write_output(
    output_path: Path,
    corpus: dict[str, Any],
    chunks: list[Chunk],
    stats: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    output = {
        "source_corpus": str(args.input),
        "root_url": corpus.get("root_url", ""),
        "processed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "parameters": {
            "min_body_chars": args.min_body_chars,
            "max_chunk_chars": args.max_chunk_chars,
        },
        "summary": stats,
        "chunks": [chunk_to_dict(c) for c in chunks],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    for i in range(1, 1000):
        candidate = path.with_name(f"{stem}-{i}{suffix}")
        if not candidate.exists():
            return candidate
    return path.with_name(f"{stem}-{int(dt.datetime.now().timestamp())}{suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Process a raw Horatio corpus JSON into pipeline-ready chunks. "
            "Filters empty sections and splits oversized ones using "
            "content-type-aware strategies."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the corpus JSON produced by capture-content.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path for the processed chunks JSON. "
            "Default: <input-dir>/chunks-YYYY-MM-DD.json"
        ),
    )
    parser.add_argument(
        "--min-body-chars",
        type=int,
        default=DEFAULT_MIN_BODY_CHARS,
        help=(
            f"Drop sections with fewer than this many body characters "
            f"(default: {DEFAULT_MIN_BODY_CHARS})"
        ),
    )
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=DEFAULT_MAX_CHUNK_CHARS,
        help=(
            f"Split sections exceeding this many characters "
            f"(default: {DEFAULT_MAX_CHUNK_CHARS})"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 2

    print(f"Reading corpus: {args.input}", flush=True)
    corpus = json.loads(args.input.read_text(encoding="utf-8"))

    chunks, stats = process_corpus(corpus, args.min_body_chars, args.max_chunk_chars)

    date_code = dt.date.today().isoformat()
    default_output = args.input.parent / f"chunks-{date_code}.json"
    output_path = unique_path(args.output or default_output)

    write_output(output_path, corpus, chunks, stats, args)

    print(f"\nCorpus processing complete:")
    print(f"  Pages:               {stats['pages']}")
    print(f"  Sections total:      {stats['sections_total']}")
    print(f"  Sections filtered:   {stats['sections_filtered']}")
    print(f"  Sections unsplit:    {stats['sections_unsplit']}")
    print(f"  Sections split:      {stats['sections_split']}")
    print(f"  Chunks produced:     {stats['chunks_total']}")
    print(f"  Content types:       {stats['content_types']}")

    if stats["code_blocks_over_limit"]:
        print(f"\n  WARNING: {len(stats['code_blocks_over_limit'])} code block(s) exceed "
              f"--max-chunk-chars and will be truncated by the embedding model:")
        for entry in stats["code_blocks_over_limit"]:
            print(f"    {entry['heading_text']} ({entry['body_chars']} chars) — {entry['source_url']}")

    print(f"\nWrote chunks: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())