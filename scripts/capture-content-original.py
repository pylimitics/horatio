#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import difflib
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


ROOT_URL = "https://securitydocs.cisco.com/docs/csa/olh/119038.dita"
DEFAULT_STORAGE_DIR = Path("data/securitydocs_weekly_changes")
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_WORKERS = 4
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 2.0
USER_AGENT = "securitydocs-weekly-change-monitor/1.0"

DITA_LINK_RE = re.compile(r"^/docs/csa/olh/\d+\.dita$")
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
WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
BLANK_LINE_RE = re.compile(r"\n{3,}")
SKIP_TEXT_PARENTS = {"script", "style", "noscript", "svg", "canvas"}
CONTENT_CONTAINER_HINTS = (
    "topic-content",
    "article-content",
    "document-content",
    "content-body",
    "main-content",
    "prose",
)
CONTENT_TAGS = {
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
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


@dataclasses.dataclass(frozen=True)
class PageSnapshot:
    url: str
    title: str
    content: str
    sha256: str


@dataclasses.dataclass(frozen=True)
class FetchFailure:
    url: str
    error: str
    attempts: int


class TocLinkParser(html.parser.HTMLParser):
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
        if href and attr_map.get("data-anchor-id") and DITA_LINK_RE.match(href):
            self.hrefs.append(href)


class BodyContentParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._in_body = False
        self._tag_stack: list[str] = []
        self._skip_depth = 0
        self._body_parts: list[str] = []
        self._content_parts: list[str] = []
        self._content_depth = 0
        self._title_parts: list[str] = []
        self._in_title = False

    @property
    def title(self) -> str:
        return normalize_text(" ".join(self._title_parts))

    @property
    def content(self) -> str:
        focused_content = normalize_content("".join(self._content_parts))
        if focused_content:
            return focused_content
        return normalize_content("".join(self._body_parts))

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
        self._append(data)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Snapshot Cisco Secure Access documentation pages from the root TOC "
            "and write a dated DITA change report."
        )
    )
    parser.add_argument(
        "--root-url",
        default=ROOT_URL,
        help=f"Root documentation URL with the TOC (default: {ROOT_URL})",
    )
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=DEFAULT_STORAGE_DIR,
        help=f"Local snapshot/report directory (default: {DEFAULT_STORAGE_DIR})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Timeout in seconds per request (default: {DEFAULT_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Concurrent page fetches (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--date-code",
        default=dt.date.today().isoformat(),
        help="Date code for snapshot/report filenames (default: today)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Limit the number of TOC pages fetched. Useful for smoke tests.",
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


def fetch_text(url: str, timeout: float) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


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


def parse_page(url: str, page_html: str) -> PageSnapshot:
    embedded_html = extract_embedded_dita_html(page_html)
    if embedded_html:
        page_html = f"<body>{embedded_html}</body>"
    parser = BodyContentParser()
    parser.feed(page_html)
    content = parser.content
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return PageSnapshot(
        url=url,
        title=parser.title or first_content_line(content),
        content=content,
        sha256=digest,
    )


def extract_embedded_dita_html(page_html: str) -> str | None:
    for pattern in EMBEDDED_DITA_PATTERNS:
        match = pattern.search(page_html)
        if match:
            return match.group("article")

    fallback_match = ARTICLE_FALLBACK_RE.search(page_html)
    if fallback_match:
        return fallback_match.group("article")
    return None


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
        except Exception as exc:  # pragma: no cover - network-specific
            last_error = exc
            if attempt == attempts:
                break
            sleep_seconds = retry_backoff * (2 ** (attempt - 1))
            time.sleep(sleep_seconds)

    assert last_error is not None
    raise RuntimeError(
        f"{last_error} (after {attempts} attempts)"
    ) from last_error


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
            except Exception as exc:  # pragma: no cover - network-specific
                failures.append(
                    FetchFailure(
                        url=url,
                        error=str(exc),
                        attempts=retries + 1,
                    )
                )
            if index == total or index % 25 == 0:
                print(f"Fetched {index}/{total} pages...", flush=True)

    snapshots.sort(key=lambda snapshot: urls.index(snapshot.url))
    failures.sort(key=lambda failure: urls.index(failure.url))
    return snapshots, failures


def normalize_content(value: str) -> str:
    lines = [normalize_text(line) for line in value.splitlines()]
    content_lines = [line for line in lines if line]
    return "\n".join(content_lines)


def normalize_text(value: str) -> str:
    value = html.unescape(value)
    value = WHITESPACE_RE.sub(" ", value)
    value = BLANK_LINE_RE.sub("\n\n", value)
    return value.strip()


def snapshot_to_json(
    root_url: str,
    date_code: str,
    snapshots: list[PageSnapshot],
    failures: list[FetchFailure],
    page_limit: int | None,
    retries: int,
    retry_backoff: float,
) -> dict[str, Any]:
    return {
        "root_url": root_url,
        "date_code": date_code,
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "page_limit": page_limit,
        "retries": retries,
        "retry_backoff": retry_backoff,
        "pages": [dataclasses.asdict(snapshot) for snapshot in snapshots],
        "errors": [dataclasses.asdict(failure) for failure in failures],
    }


def load_snapshot(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_snapshot(path: Path, snapshot: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def find_previous_snapshot(
    snapshot_dir: Path,
    current_snapshot: Path,
    root_url: str,
    page_limit: int | None,
) -> Path | None:
    candidates = sorted(snapshot_dir.glob("snapshot-*.json"))
    previous = [
        path
        for path in candidates
        if path != current_snapshot
        and snapshot_matches_scope(path, root_url, page_limit)
    ]
    return previous[-1] if previous else None


def snapshot_matches_scope(
    snapshot_path: Path,
    root_url: str,
    page_limit: int | None,
) -> bool:
    snapshot = load_snapshot(snapshot_path)
    if not snapshot:
        return False
    return (
        snapshot.get("root_url") == root_url
        and snapshot.get("page_limit") == page_limit
    )


def build_report(
    root_url: str,
    date_code: str,
    previous_snapshot_path: Path | None,
    previous_snapshot: dict[str, Any] | None,
    current_snapshot: dict[str, Any],
) -> str:
    previous_pages = {
        page["url"]: page for page in (previous_snapshot or {}).get("pages", [])
    }
    current_pages = {page["url"]: page for page in current_snapshot["pages"]}

    changed_urls = [
        url
        for url, page in current_pages.items()
        if previous_pages.get(url, {}).get("sha256") != page["sha256"]
    ]
    removed_urls = [
        url for url in previous_pages if url not in current_pages
    ]
    added_urls = [
        url for url in current_pages if url not in previous_pages
    ]

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE reference PUBLIC "-//CISCO//DTD DITA 1.3 Reference v1.0//EN" "cisco-reference.dtd"[]>',
        f'<reference id="securitydocs-weekly-changes-{xml_id(date_code)}" xml:lang="en_US">',
        f"  <title>Cisco Secure Access Documentation Changes {escape_xml(date_code)}</title>",
        "  <refbody>",
        "    <section>",
        "      <title>Summary</title>",
        "      <ul>",
        f"        <li><p>Document URL: <xref href=\"{escape_xml(root_url)}\" format=\"html\" scope=\"external\">{escape_xml(root_url)}</xref></p></li>",
        f"        <li><p>Snapshot date: {escape_xml(date_code)}</p></li>",
        f"        <li><p>Previous snapshot: {escape_xml(previous_snapshot_date(previous_snapshot_path))}</p></li>",
        f"        <li><p>Pages checked: {len(current_pages)}</p></li>",
        f"        <li><p>Pages changed: {len(changed_urls)}</p></li>",
        f"        <li><p>Pages added: {len(added_urls)}</p></li>",
        f"        <li><p>Pages removed: {len(removed_urls)}</p></li>",
        "      </ul>",
        "    </section>",
    ]

    errors = current_snapshot.get("errors", {})
    if errors:
        lines.extend(
            [
                "    <section>",
                "      <title>Fetch Errors</title>",
                "      <ul>",
            ]
        )
        normalized_errors = normalize_error_entries(errors)
        for entry in normalized_errors:
            url = entry["url"]
            error = entry["error"]
            attempts = entry.get("attempts")
            details = error
            if attempts:
                details = f"{error}; attempts: {attempts}"
            lines.append(
                f"        <li><p><xref href=\"{escape_xml(url)}\" format=\"html\" scope=\"external\">{escape_xml(url)}</xref>: {escape_xml(details)}</p></li>"
            )
        lines.extend(["      </ul>", "    </section>"])

    if not previous_snapshot:
        lines.extend(
            [
                "    <section>",
                "      <title>Baseline Created</title>",
                "      <p>No previous snapshot was available, so this run created the baseline for future comparisons.</p>",
                "    </section>",
            ]
        )
    elif not changed_urls and not removed_urls:
        lines.extend(
            [
                "    <section>",
                "      <title>No Content Changes</title>",
                "      <p>No paragraph or line changes were detected in the scanned body content.</p>",
                "    </section>",
            ]
        )
    else:
        lines.extend(
            [
                "    <section>",
                "      <title>Changed Pages</title>",
                "    </section>",
            ]
        )
        for url in changed_urls:
            current_page = current_pages[url]
            previous_page = previous_pages.get(url)
            title = current_page.get("title") or url
            lines.extend(
                [
                    "    <section>",
                    f"      <title>{escape_xml(title)}</title>",
                    f"      <p><xref href=\"{escape_xml(url)}\" format=\"html\" scope=\"external\">{escape_xml(url)}</xref></p>",
                ]
            )
            if previous_page is None:
                lines.append("      <p>New page in the TOC.</p>")
            else:
                diff_entries = make_diff_entries(previous_page, current_page)
                if diff_entries:
                    lines.append("      <ul>")
                    for entry in diff_entries:
                        label = "Added" if entry["kind"] == "added" else "Removed"
                        lines.append(
                            f"        <li><p>{label}: {escape_xml(entry['text'])}</p></li>"
                        )
                    lines.append("      </ul>")
                else:
                    lines.append("      <p>Content changed, but no line-level delta was produced.</p>")
            lines.append("    </section>")

        if removed_urls:
            lines.extend(
                [
                    "    <section>",
                    "      <title>Removed Pages</title>",
                    "      <ul>",
                ]
            )
            for url in removed_urls:
                lines.append(
                    f"        <li><p><xref href=\"{escape_xml(url)}\" format=\"html\" scope=\"external\">{escape_xml(url)}</xref></p></li>"
                )
            lines.extend(["      </ul>", "    </section>"])

    lines.extend(["  </refbody>", "</reference>", ""])
    return "\n".join(lines)


def make_diff_entries(
    previous_page: dict[str, Any],
    current_page: dict[str, Any],
) -> list[dict[str, str]]:
    previous_lines = previous_page["content"].splitlines()
    current_lines = current_page["content"].splitlines()
    matcher = difflib.SequenceMatcher(a=previous_lines, b=current_lines)
    entries: list[dict[str, str]] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag in {"replace", "delete"}:
            for line in previous_lines[i1:i2]:
                if line:
                    entries.append({"kind": "removed", "text": line})
        if tag in {"replace", "insert"}:
            for line in current_lines[j1:j2]:
                if line:
                    entries.append({"kind": "added", "text": line})

    return entries


def escape_xml(value: object) -> str:
    return html.escape(str(value), quote=True)


def xml_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-")


def normalize_error_entries(errors: Any) -> list[dict[str, Any]]:
    if isinstance(errors, dict):
        return [
            {"url": url, "error": error}
            for url, error in sorted(errors.items())
        ]
    if isinstance(errors, list):
        return errors
    return []


def previous_snapshot_date(previous_snapshot_path: Path | None) -> str:
    if previous_snapshot_path is None:
        return "none"
    match = re.search(r"snapshot-(\d{4}-\d{2}-\d{2})(?:-\d+)?\.json$", previous_snapshot_path.name)
    if match:
        return match.group(1)
    return previous_snapshot_path.name


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


def main() -> int:
    args = parse_args()
    storage_dir: Path = args.storage_dir
    snapshot_dir = storage_dir / "snapshots"
    report_dir = storage_dir / "reports"
    date_code: str = args.date_code

    snapshot_path = unique_path(snapshot_dir / f"snapshot-{date_code}.json")
    report_path = unique_path(report_dir / f"securitydocs-changes-{date_code}.dita")

    print(f"Fetching root TOC: {args.root_url}", flush=True)
    try:
        root_html = fetch_text(args.root_url, args.timeout)
    except urllib.error.URLError as exc:
        print(f"Failed to fetch root URL: {exc}", file=sys.stderr)
        return 2

    urls = extract_toc_urls(args.root_url, root_html)
    if not urls:
        print("No TOC DITA links found in the root page.", file=sys.stderr)
        return 2
    total_urls = len(urls)
    if args.max_pages is not None:
        urls = urls[: args.max_pages]

    if args.max_pages is None:
        print(f"Found {len(urls)} TOC pages. Fetching page content...", flush=True)
    else:
        print(
            f"Found {total_urls} TOC pages. Fetching first {len(urls)} pages...",
            flush=True,
        )
    snapshots, failures = fetch_snapshots(
        urls,
        args.timeout,
        args.workers,
        args.retries,
        args.retry_backoff,
    )
    if failures:
        print(
            f"Completed fetches with {len(failures)} page failures after retries.",
            flush=True,
        )
    current_snapshot = snapshot_to_json(
        args.root_url,
        date_code,
        snapshots,
        failures,
        args.max_pages,
        args.retries,
        args.retry_backoff,
    )

    previous_snapshot_path = find_previous_snapshot(
        snapshot_dir,
        snapshot_path,
        args.root_url,
        args.max_pages,
    )
    previous_snapshot = (
        load_snapshot(previous_snapshot_path) if previous_snapshot_path else None
    )
    report = build_report(
        root_url=args.root_url,
        date_code=date_code,
        previous_snapshot_path=previous_snapshot_path,
        previous_snapshot=previous_snapshot,
        current_snapshot=current_snapshot,
    )

    write_snapshot(snapshot_path, current_snapshot)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print(f"Wrote snapshot: {snapshot_path}")
    print(f"Wrote DITA report: {report_path}")
    if failures:
        print(f"Completed with {len(failures)} fetch errors.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
