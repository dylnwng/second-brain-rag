"""Heading-aware chunking.

Notes are split at markdown headings first, so every chunk belongs to exactly one section
and can be cited as `path > heading path`. Sections longer than the target are packed
paragraph by paragraph, with a small overlap carried between consecutive chunks of the
same section. Overlap never crosses a heading: a new section is a new topic.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from sbrag.parse import render_wikilinks

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def estimate_tokens(text: str) -> int:
    """~4 characters per token. An estimate for sizing chunks only; billed token counts
    come from the embedding API response, and the ingestion log says which one it used."""
    return math.ceil(len(text) / 4) if text else 0


@dataclass
class Chunk:
    ordinal: int
    heading_path: str
    content: str
    token_estimate: int


@dataclass
class Section:
    heading_path: str
    text: str


def split_sections(body: str, title: str) -> list[Section]:
    """Split on headings outside code fences. Text before the first heading is filed under
    the note title."""
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    in_fence = False

    def path() -> str:
        return " > ".join(t for _, t in stack) if stack else title

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            sections.append(Section(path(), text))
        buf.clear()

    for line in body.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
            buf.append(line)
            continue
        m = None if in_fence else HEADING_RE.match(line)
        if m:
            flush()
            level, text = len(m.group(1)), render_wikilinks(m.group(2)).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
        else:
            buf.append(line)
    flush()
    return sections


def _blocks(text: str) -> list[str]:
    """Paragraphs separated by blank lines; a fenced code block stays one block."""
    blocks: list[str] = []
    cur: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            if cur:
                blocks.append("\n".join(cur))
                cur = []
        else:
            cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    return blocks


def _split_oversize(block: str, target: int) -> list[str]:
    """Break a single block bigger than the target: by lines (tables, lists), then by
    sentences, then by words as a last resort."""
    if estimate_tokens(block) <= target:
        return [block]
    for splitter, joiner in ((lambda s: s.splitlines(), "\n"),
                             (lambda s: SENTENCE_RE.split(s), " "),
                             (lambda s: s.split(" "), " ")):
        parts = [p for p in splitter(block) if p.strip()]
        if len(parts) > 1:
            out: list[str] = []
            cur: list[str] = []
            for p in parts:
                if cur and estimate_tokens(joiner.join(cur + [p])) > target:
                    out.append(joiner.join(cur))
                    cur = []
                cur.append(p)
            if cur:
                out.append(joiner.join(cur))
            # Recurse in case a single line/sentence is still too long.
            return [piece for o in out for piece in _split_oversize(o, target)]
    # One unbreakable token run (e.g. a giant URL): hard cut by characters.
    step = target * 4
    return [block[i:i + step] for i in range(0, len(block), step)]


def _tail(text: str, overlap: int) -> str:
    """Last ~`overlap` tokens of text, cut on a word boundary."""
    if overlap <= 0:
        return ""
    words = text.split()
    out: list[str] = []
    for w in reversed(words):
        if estimate_tokens(" ".join([w] + out)) > overlap:
            break
        out.insert(0, w)
    return " ".join(out)


def chunk_section(section: Section, target: int, overlap: int) -> list[str]:
    pieces = [p for b in _blocks(section.text) for p in _split_oversize(b, target)]
    chunks: list[str] = []
    cur: list[str] = []
    for p in pieces:
        if cur and estimate_tokens("\n\n".join(cur + [p])) > target:
            chunks.append("\n\n".join(cur))
            carry = _tail(chunks[-1], overlap)
            cur = [carry] if carry else []
            # If carry + piece would still overflow, drop the carry rather than exceed target.
            if cur and estimate_tokens("\n\n".join(cur + [p])) > target:
                cur = []
        cur.append(p)
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def chunk_note(body: str, title: str, target: int = 500, overlap: int = 50) -> list[Chunk]:
    if overlap >= target:
        raise ValueError("overlap must be smaller than target")
    out: list[Chunk] = []
    for section in split_sections(body, title):
        for text in chunk_section(section, target, overlap):
            text = render_wikilinks(text).strip()
            if not text:
                continue
            out.append(Chunk(len(out), section.heading_path, text, estimate_tokens(text)))
    return out


def embed_input(title: str, chunk: Chunk) -> str:
    """Document-side input in the format gemini-embedding-2 recommends for retrieval
    ("title: {title} | text: {content}"). The heading path goes in the title so a
    chunk like "Week 3" still knows it belongs to "Fitness Plan"."""
    hp = chunk.heading_path
    full_title = hp if hp.startswith(title) else f"{title} > {hp}"
    return f"title: {full_title} | text: {chunk.content}"
