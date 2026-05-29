#!/usr/bin/env python3
"""Capture Security Cloud Control documentation content for the Horatio
neuro-symbolic pipeline.

Fetches every topic page listed in a book's TOC, extracts body text and
heading structure, splits each page into headed sections, and writes a
JSON snapshot suitable for downstream chunking, embedding, and graph
population.

Derived from securitydocs_weekly_changes.py (Cisco Secure Access weekly
change monitor).  The change-detection and DITA report functionality has
been removed; heading-aware parsing and section splitting have been added.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import hashlib
import html
import html.parser
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any
import ssl
import certifi
# Override the default SSL context to use certifi's bundle
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Root TOC page for the book you want to capture.  Override with --root-url.
# Default targets the Security Cloud Control Getting Started Guide.
ROOT_URL = "https://securitydocs.cisco.com/docs/scc/gsg/new/106117.dita"

DEFAULT_OUTPUT_DIR = Path("data/horatio")
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_WORKERS = 4
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 2.0

# Identifies this client to the server.
USER_AGENT = "horatio-content-capture/1.0"

# Matches any SCC documentation topic URL regardless of book or section.
# The CSA original matched only /docs/csa/olh/\d+\.dita; this is intentionally
# broader so the same script works across SCC books by changing only ROOT_URL.
DITA_LINK_RE = re.compile(r"^/docs/scc/[\w/-]+/\d+\.dita$")

# ---------------------------------------------------------------------------
# Patterns for extracting embedded DITA content from the page's <script> block.
# The portal injects body HTML into a JS variable rather than rendering it
# server-side, so plain HTTP fetches return the full content without needing
# a headless browser.
# ---------------------------------------------------------------------------

EMBEDDED_DITA_PATTERNS = (
    re.compile(
        r"window\.__dita\$html\s*=\s*`(?P<article>.*?)`;",
        re.DOTALL,
    ),
    re.compile(
        r"ditaContentDiv\.innerHTML\s*=\s*`(?P<article>.*?)`;",
        re.DOTALL,
    ),
)

ARTICLE_FALLBACK_RE = re.compile(
    r"(?P<article><article\b[^>]*class=\"[^\"]*\btopic\b[^\"]*\"[^>]*>.*?</article>)",
    re.DOTALL,
)

# ---------------------------------------------------------------------------
# Text normalisation patterns
# ---------------------------------------------------------------------------

WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
BLANK_LINE_RE = re.compile(r"\n{3,}")

# ---------------------------------------------------------------------------
# HTML parsing helpers
# ---------------------------------------------------------------------------

SKIP_TEXT_PARENTS = {"script", "style", "noscript", "svg", "canvas"}

CONTENT_CONTAINER_HINTS = (
    "topic-content",
    "article-content",
    "document-content",
    "content-body",
    "main-content",
    "prose",
)

HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

CONTENT_TAGS = HEADING_TAGS | {
    "p",
    "li",
    "dt",
    "dd",
    "figcaption",
    "caption",
    "th",
    "td",
    "pre",
    "code",
}

BLOCK_TAGS = CONTENT_TAGS | {
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "div",
    "dl",
    "figure",
    "footer",
    "header",
    "hr",
    "main",
    "nav",
    "ol",
    "section",
    "table",
    "tbody",
    "thead",
    "tfoot",
    "tr",
    "ul",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Heading:
    """A single heading encountered in the page body."""
    level: int          # 1–6, corresponding to h1–h6
    text: str           # Normalised heading text
    # Character offset in the flat content string where this heading begins.
    # Used by split_into_sections to slice the content.
    content_offset: int


@dataclasses.dataclass
class Section:
    """A headed section of a page: one heading plus the prose beneath it.

    This is the primary unit consumed by the embedding and graph pipeline.
    Each Section maps to one vector-store chunk and one candidate graph node.
    """
    heading_level: int
    heading_text: str
    # Breadcrumb of ancestor heading texts from h1 down to this section's
    # parent, not including this section's own heading.  Provides context
    # when the section is retrieved in isolation.
    heading_path: list[str]
    # The prose content beneath this heading, not including the heading line
    # itself.  May be empty for container headings that hold only sub-sections.
    body: str
    # Full URL of the source page, used for citations.
    source_url: str
    # Title of the source page (usually the h1 or <title> element).
    source_title: str


@dataclasses.dataclass
class PageSnapshot:
    """Complete parsed representation of one documentation topic page."""
    url: str
    title: str
    # Flat normalised text of the entire page body (headings included).
    # Preserved for compatibility and full-page search use cases.
    content: str
    # SHA-256 of content, for change detection on re-runs.
    sha256: str
    # Ordered list of headings found in the page, with their levels.
    headings: list[Heading]
    # Content split into headed sections.  The preferred unit for the pipeline.
    sections: list[Section]


@dataclasses.dataclass(frozen=True)
class FetchFailure:
    url: str
    error: str
    attempts: int


# ---------------------------------------------------------------------------
# HTML parsers
# ---------------------------------------------------------------------------

class TocLinkParser(html.parser.HTMLParser):
    """Extracts topic URLs from a book's TOC page.

    Collects href values from <a> tags whose href matches DITA_LINK_RE.
    The data-anchor-id filter used in the CSA original is omitted here
    because SCC TOC link attributes have not yet been confirmed; the
    DITA_LINK_RE pattern provides sufficient specificity.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag.lower() != "a":
            return
        attr_map = dict(attrs)
        href = attr_map.get("href")
        if href and DITA_LINK_RE.match(href):
            self.hrefs.append(href)


class BodyContentParser(html.parser.HTMLParser):
    """Parses a documentation topic page and extracts structured content.

    Produces:
      - title: the page <title> element text
      - content: flat normalised body text (headings and prose interleaved)
      - headings: ordered list of Heading objects with level, text, and
        their character offset within `content`

    The heading offsets are used downstream by split_into_sections() to
    slice the flat content string into per-section bodies without re-parsing.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._in_body = False
        self._tag_stack: list[str] = []
        self._skip_depth = 0

        # Accumulates all body text (both focused-container and fallback paths)
        self._body_parts: list[str] = []
        # Accumulates text only when inside a recognised content container
        self._content_parts: list[str] = []
        self._content_depth = 0

        self._title_parts: list[str] = []
        self._in_title = False

        # Heading tracking
        self._in_heading: str | None = None   # current heading tag, e.g. "h2"
        self._heading_parts: list[str] = []   # text accumulator for current heading
        self._headings: list[Heading] = []

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def title(self) -> str:
        return normalize_text(" ".join(self._title_parts))

    @property
    def content(self) -> str:
        focused = normalize_content("".join(self._content_parts))
        if focused:
            return focused
        return normalize_content("".join(self._body_parts))

    @property
    def headings(self) -> list[Heading]:
        return self._headings

    # ------------------------------------------------------------------
    # HTMLParser callbacks
    # ------------------------------------------------------------------

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        tag = tag.lower()

        if tag == "title":
            self._in_title = True
            return

        if tag == "body":
            self._in_body = True
            self._tag_stack.append(tag)
            return

        if not self._in_body:
            return

        if tag in SKIP_TEXT_PARENTS:
            self._skip_depth += 1

        attr_map = {name.lower(): value or "" for name, value in attrs}
        if self._content_depth == 0 and self._looks_like_content_container(
            tag, attr_map
        ):
            self._content_depth = 1
        elif self._content_depth > 0:
            self._content_depth += 1

        if tag in BLOCK_TAGS:
            self._append("\n")

        # Begin heading accumulation.  Record the character offset *after* the
        # newline we just appended so the heading text starts the line cleanly.
        if tag in HEADING_TAGS:
            self._in_heading = tag
            self._heading_parts = []

        self._tag_stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        if tag == "title":
            self._in_title = False
            return

        if not self._in_body:
            return

        if tag in BLOCK_TAGS:
            self._append("\n")

        if tag in SKIP_TEXT_PARENTS and self._skip_depth > 0:
            self._skip_depth -= 1

        if self._content_depth > 0:
            self._content_depth -= 1

        # Finalise a heading: compute its offset and store it.
        if tag in HEADING_TAGS and self._in_heading == tag:
            heading_text = normalize_text("".join(self._heading_parts))
            if heading_text:
                # Offset is measured in the *focused* content stream when
                # available, falling back to the full body stream.  We compute
                # it lazily in split_into_sections() from the flat content
                # string, so here we store a sentinel and resolve it after
                # parsing completes.
                self._headings.append(
                    Heading(
                        level=int(tag[1]),
                        text=heading_text,
                        content_offset=-1,   # resolved in resolve_heading_offsets()
                    )
                )
            self._in_heading = None
            self._heading_parts = []

        if tag == "body":
            self._in_body = False
            self._tag_stack.clear()
            self._content_depth = 0
            return

        if self._tag_stack:
            self._tag_stack.pop()

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_parts.append(data)
            return

        if not self._in_body or self._skip_depth > 0:
            return

        if self._content_depth == 0 and not self._is_in_content_tag():
            return

        if self._in_heading:
            self._heading_parts.append(data)

        self._append(data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append(self, value: str) -> None:
        self._body_parts.append(value)
        if self._content_depth > 0:
            self._content_parts.append(value)

    def _is_in_content_tag(self) -> bool:
        return any(tag in CONTENT_TAGS for tag in self._tag_stack)

    def _looks_like_content_container(
        self, tag: str, attrs: dict[str, str]
    ) -> bool:
        if tag in {"main", "article"}:
            return True
        names = " ".join(
            value
            for key, value in attrs.items()
            if key in {"id", "class", "role", "data-testid"}
        ).lower()
        return any(hint in names for hint in CONTENT_CONTAINER_HINTS)


# ---------------------------------------------------------------------------
# Heading offset resolution and section splitting
# ---------------------------------------------------------------------------

def resolve_heading_offsets(content: str, headings: list[Heading]) -> None:
    """Populate content_offset on each Heading by searching the flat content.

    Modifies headings in place.  Searches forward from the previous heading's
    offset to handle duplicate heading texts correctly.
    """
    search_start = 0
    for heading in headings:
        idx = content.find(heading.text, search_start)
        if idx == -1:
            # Heading text not found verbatim (possible after normalisation).
            # Leave offset at -1; split_into_sections will skip it.
            continue
        heading.content_offset = idx
        search_start = idx + len(heading.text)


def split_into_sections(
    page: "PageSnapshot",
) -> list[Section]:
    """Split a page's flat content into headed sections.

    Each section runs from one heading to the start of the next heading at
    the same or higher level (lower number), or to end-of-content.

    Returns one Section per heading.  Pages with no headings return a single
    Section with an empty heading (the whole page body as an unheaded block).
    """
    content = page.content
    headings = [h for h in page.headings if h.content_offset >= 0]

    if not headings:
        return [
            Section(
                heading_level=0,
                heading_text="",
                heading_path=[],
                body=content.strip(),
                source_url=page.url,
                source_title=page.title,
            )
        ]

    sections: list[Section] = []
    # Stack tracks the current ancestor path: list of (level, text) tuples.
    ancestor_stack: list[tuple[int, str]] = []

    for i, heading in enumerate(headings):
        # Determine where this section's body ends.
        next_offset = headings[i + 1].content_offset if i + 1 < len(headings) else len(content)

        # The body is the content between the end of the heading text and the
        # start of the next heading.
        body_start = heading.content_offset + len(heading.text)
        body = normalize_content(content[body_start:next_offset])

        # Update ancestor stack: pop headings at same or deeper level.
        while ancestor_stack and ancestor_stack[-1][0] >= heading.level:
            ancestor_stack.pop()

        heading_path = [text for _, text in ancestor_stack]

        sections.append(
            Section(
                heading_level=heading.level,
                heading_text=heading.text,
                heading_path=heading_path,
                body=body,
                source_url=page.url,
                source_title=page.title,
            )
        )

        # Push this heading onto the ancestor stack for subsequent sections.
        ancestor_stack.append((heading.level, heading.text))

    return sections


# ---------------------------------------------------------------------------
# Page fetching and parsing
# ---------------------------------------------------------------------------

def fetch_text(url: str, timeout: float) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def extract_embedded_dita_html(page_html: str) -> str | None:
    """Extract the DITA body HTML injected into the page's <script> block."""
    for pattern in EMBEDDED_DITA_PATTERNS:
        match = pattern.search(page_html)
        if match:
            return match.group("article")
    fallback_match = ARTICLE_FALLBACK_RE.search(page_html)
    if fallback_match:
        return fallback_match.group("article")
    return None


def parse_page(url: str, page_html: str) -> PageSnapshot:
    embedded_html = extract_embedded_dita_html(page_html)
    if embedded_html:
        page_html = f"<body>{embedded_html}</body>"

    parser = BodyContentParser()
    parser.feed(page_html)

    content = parser.content
    headings = parser.headings
    resolve_heading_offsets(content, headings)

    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()

    # Build a preliminary snapshot without sections so split_into_sections
    # can reference page.url and page.title.
    snapshot = PageSnapshot(
        url=url,
        title=parser.title or first_content_line(content),
        content=content,
        sha256=digest,
        headings=headings,
        sections=[],
    )
    snapshot.sections = split_into_sections(snapshot)
    return snapshot


def first_content_line(content: str) -> str:
    for line in content.splitlines():
        if line:
            return line
    return ""


def fetch_page_snapshot(url: str, timeout: float) -> PageSnapshot:
    return parse_page(url, fetch_text(url, timeout))


def fetch_page_snapshot_with_retries(
    url: str,
    timeout: float,
    retries: int,
    retry_backoff: float,
) -> PageSnapshot:
    attempts = retries + 1
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            return fetch_page_snapshot(url, timeout)
        except Exception as exc:
            last_error = exc
            if attempt == attempts:
                break
            sleep_seconds = retry_backoff * (2 ** (attempt - 1))
            time.sleep(sleep_seconds)

    assert last_error is not None
    raise RuntimeError(f"{last_error} (after {attempts} attempts)") from last_error


# ---------------------------------------------------------------------------
# TOC extraction
# ---------------------------------------------------------------------------

def extract_toc_urls(root_url: str, page_html: str) -> list[str]:
    parser = TocLinkParser()
    parser.feed(page_html)

    seen: set[str] = set()
    urls: list[str] = []
    for href in parser.hrefs:
        url = urllib.parse.urljoin(root_url, href)
        url, _fragment = urllib.parse.urldefrag(url)
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


# ---------------------------------------------------------------------------
# Concurrent fetching
# ---------------------------------------------------------------------------

def fetch_snapshots(
    urls: list[str],
    timeout: float,
    workers: int,
    retries: int,
    retry_backoff: float,
) -> tuple[list[PageSnapshot], list[FetchFailure]]:
    snapshots: list[PageSnapshot] = []
    failures: list[FetchFailure] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_url = {
            executor.submit(
                fetch_page_snapshot_with_retries,
                url,
                timeout,
                retries,
                retry_backoff,
            ): url
            for url in urls
        }
        total = len(future_to_url)
        for index, future in enumerate(
            concurrent.futures.as_completed(future_to_url), start=1
        ):
            url = future_to_url[future]
            try:
                snapshots.append(future.result())
            except Exception as exc:
                failures.append(
                    FetchFailure(url=url, error=str(exc), attempts=retries + 1)
                )
            if index == total or index % 25 == 0:
                print(f"Fetched {index}/{total} pages...", flush=True)

    snapshots.sort(key=lambda s: urls.index(s.url))
    failures.sort(key=lambda f: urls.index(f.url))
    return snapshots, failures


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def normalize_content(value: str) -> str:
    lines = [normalize_text(line) for line in value.splitlines()]
    return "\n".join(line for line in lines if line)


def normalize_text(value: str) -> str:
    value = html.unescape(value)
    value = WHITESPACE_RE.sub(" ", value)
    value = BLANK_LINE_RE.sub("\n\n", value)
    return value.strip()


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def section_to_dict(section: Section) -> dict[str, Any]:
    return {
        "heading_level": section.heading_level,
        "heading_text": section.heading_text,
        "heading_path": section.heading_path,
        "body": section.body,
        "source_url": section.source_url,
        "source_title": section.source_title,
    }


def heading_to_dict(heading: Heading) -> dict[str, Any]:
    return {
        "level": heading.level,
        "text": heading.text,
        "content_offset": heading.content_offset,
    }


def snapshot_to_dict(snapshot: PageSnapshot) -> dict[str, Any]:
    return {
        "url": snapshot.url,
        "title": snapshot.title,
        "content": snapshot.content,
        "sha256": snapshot.sha256,
        "headings": [heading_to_dict(h) for h in snapshot.headings],
        "sections": [section_to_dict(s) for s in snapshot.sections],
    }


def corpus_to_json(
    root_url: str,
    snapshots: list[PageSnapshot],
    failures: list[FetchFailure],
    page_limit: int | None,
    retries: int,
    retry_backoff: float,
) -> dict[str, Any]:
    total_sections = sum(len(s.sections) for s in snapshots)
    return {
        "root_url": root_url,
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "page_limit": page_limit,
        "retries": retries,
        "retry_backoff": retry_backoff,
        "summary": {
            "pages_captured": len(snapshots),
            "pages_failed": len(failures),
            "total_sections": total_sections,
        },
        "pages": [snapshot_to_dict(s) for s in snapshots],
        "errors": [dataclasses.asdict(f) for f in failures],
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(1, 1000):
        candidate = path.with_name(f"{stem}-{index}{suffix}")
        if not candidate.exists():
            return candidate
    timestamp = int(time.time())
    return path.with_name(f"{stem}-{timestamp}{suffix}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture Security Cloud Control documentation for the Horatio "
            "neuro-symbolic pipeline.  Fetches every topic page listed in a "
            "book TOC, parses heading structure, splits into sections, and "
            "writes a JSON corpus file."
        )
    )
    parser.add_argument(
        "--root-url",
        default=ROOT_URL,
        help=f"Root TOC page URL for the book to capture (default: {ROOT_URL})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for the output corpus JSON (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help=(
            "Output filename (default: corpus-YYYY-MM-DD.json in --output-dir). "
            "If the file already exists a numeric suffix is appended."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Per-request timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Concurrent page fetches (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Limit the number of topic pages fetched.  Useful for smoke tests.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"Retries per page after the initial attempt (default: {DEFAULT_RETRIES})",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=DEFAULT_BACKOFF_SECONDS,
        help=(
            "Base seconds for exponential backoff between retries "
            f"(default: {DEFAULT_BACKOFF_SECONDS})"
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    date_code = dt.date.today().isoformat()
    filename = args.output_file or f"corpus-{date_code}.json"
    output_path = unique_path(args.output_dir / filename)

    print(f"Fetching root TOC: {args.root_url}", flush=True)
    try:
        root_html = fetch_text(args.root_url, args.timeout)
    except urllib.error.URLError as exc:
        print(f"Failed to fetch root URL: {exc}", file=sys.stderr)
        return 2

    all_urls = extract_toc_urls(args.root_url, root_html)
    if not all_urls:
        print("No TOC DITA links found in the root page.", file=sys.stderr)
        return 2

    urls = all_urls[: args.max_pages] if args.max_pages is not None else all_urls
    if args.max_pages is not None:
        print(
            f"Found {len(all_urls)} TOC pages. "
            f"Fetching first {len(urls)} (--max-pages={args.max_pages})...",
            flush=True,
        )
    else:
        print(f"Found {len(urls)} TOC pages. Fetching content...", flush=True)

    snapshots, failures = fetch_snapshots(
        urls, args.timeout, args.workers, args.retries, args.retry_backoff
    )

    if failures:
        print(
            f"Completed with {len(failures)} fetch failure(s) after retries.",
            flush=True,
        )

    corpus = corpus_to_json(
        root_url=args.root_url,
        snapshots=snapshots,
        failures=failures,
        page_limit=args.max_pages,
        retries=args.retries,
        retry_backoff=args.retry_backoff,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(corpus, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = corpus["summary"]
    print(f"\nCaptured {summary['pages_captured']} pages, "
          f"{summary['total_sections']} sections, "
          f"{summary['pages_failed']} failures.")
    print(f"Wrote corpus: {output_path}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
