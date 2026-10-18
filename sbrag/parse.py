"""Parse a markdown note into structured metadata (frontmatter) and body text."""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.DOTALL)
# [[target]], [[target|alias]], [[target#heading|alias]]; ![[embeds]] are matched too.
WIKILINK_RE = re.compile(r"!?\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]+))?\]\]")
H1_RE = re.compile(r"^#\s+(.+?)\s*#*\s*$", re.MULTILINE)

# Frontmatter keys we treat as the note's creation date, in priority order.
CREATED_KEYS = ("created", "date", "created_at")
UPDATED_KEYS = ("updated", "modified", "updated_at")


@dataclass
class Note:
    path: str  # relative to vault root, posix separators
    title: str
    folder: str
    tags: list[str]
    links: list[str]
    frontmatter: dict
    body: str  # markdown body with frontmatter removed (wikilinks left intact)
    content_hash: str
    created_at: dt.datetime | None = None
    updated_at: dt.datetime | None = None
    date_source: str = "none"
    extra: dict = field(default_factory=dict)


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Return (frontmatter dict, body). Malformed YAML yields an empty dict, not a crash."""
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    try:
        data = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    return data, text[m.end():]


def normalize_tags(raw) -> list[str]:
    """Accept `tags: [a, b]`, `tags: a, b`, `tags: "#a #b"`, or a YAML list; lowercase, dedupe."""
    if raw is None:
        return []
    if isinstance(raw, str):
        items = re.split(r"[,\s]+", raw)
    elif isinstance(raw, (list, tuple)):
        items = [str(x) for x in raw]
    else:
        items = [str(raw)]
    out: list[str] = []
    for t in items:
        t = t.strip().lstrip("#").lower()
        if t and t not in out:
            out.append(t)
    return out


def _as_datetime(value) -> dt.datetime | None:
    if isinstance(value, dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=dt.UTC)
    if isinstance(value, dt.date):
        return dt.datetime(value.year, value.month, value.day, tzinfo=dt.UTC)
    if isinstance(value, str):
        try:
            return _as_datetime(dt.datetime.fromisoformat(value.strip()))
        except ValueError:
            return None
    return None


def frontmatter_date(fm: dict, keys: tuple[str, ...]) -> dt.datetime | None:
    for k in keys:
        if k in fm:
            d = _as_datetime(fm[k])
            if d:
                return d
    return None


def extract_links(body: str) -> list[str]:
    out: list[str] = []
    for m in WIKILINK_RE.finditer(body):
        target = m.group(1).strip()
        if target and target not in out:
            out.append(target)
    return out


def render_wikilinks(text: str) -> str:
    """[[target|alias]] -> alias, [[target]] -> target. Used for embedding and FTS input."""
    return WIKILINK_RE.sub(lambda m: (m.group(2) or m.group(1)).strip(), text)


def note_title(fm: dict, body: str, path: Path) -> str:
    if isinstance(fm.get("title"), str) and fm["title"].strip():
        return fm["title"].strip()
    m = H1_RE.search(body)
    if m:
        return render_wikilinks(m.group(1)).strip()
    return path.stem.replace("-", " ").replace("_", " ")


def parse_note(text: str, rel_path: str) -> Note:
    fm, body = split_frontmatter(text)
    p = Path(rel_path)
    folder = "" if str(p.parent) == "." else p.parent.as_posix()
    created = frontmatter_date(fm, CREATED_KEYS)
    return Note(
        path=p.as_posix(),
        title=note_title(fm, body, p),
        folder=folder,
        tags=normalize_tags(fm.get("tags")),
        links=extract_links(body),
        frontmatter=_jsonable(fm),
        body=body,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        created_at=created,
        updated_at=frontmatter_date(fm, UPDATED_KEYS),
        date_source="frontmatter" if created else "none",
    )


def _jsonable(obj):
    """YAML can produce dates; jsonb can't store them. Stringify anything non-JSON."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (dt.date, dt.datetime)):
        return obj.isoformat()
    return str(obj)
