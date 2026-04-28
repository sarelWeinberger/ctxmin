from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_CHUNK_OVERLAP_LINES, DEFAULT_CHUNK_TOKEN_LIMIT, DEFAULT_LINE_WINDOW
from .utils import estimate_tokens, language_for_path, sha256_text


@dataclass
class Chunk:
    chunk_id: str
    repo_path: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    symbol_name: str | None
    chunk_type: str
    text: str
    token_estimate: int
    content_hash: str
    embedding: object | None = None
    cache_key: str | None = None


def make_chunk(
    repo_path: str,
    file_path: str,
    language: str,
    lines: list[str],
    start_line: int,
    end_line: int,
    symbol_name: str | None = None,
    chunk_type: str = "unknown",
) -> Chunk:
    text = "\n".join(lines[start_line - 1 : end_line])
    content_hash = sha256_text(text)
    chunk_id = sha256_text(f"{repo_path}:{file_path}:{start_line}:{end_line}:{content_hash}")[:24]
    return Chunk(
        chunk_id=chunk_id,
        repo_path=repo_path,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        language=language,
        symbol_name=symbol_name,
        chunk_type=chunk_type,
        text=text,
        token_estimate=estimate_tokens(text),
        content_hash=content_hash,
    )


def _split_long_chunk(chunk: Chunk, max_tokens: int, overlap_lines: int) -> list[Chunk]:
    if chunk.token_estimate <= max_tokens:
        return [chunk]
    lines = chunk.text.splitlines()
    if not lines:
        return [chunk]
    approx_lines = max(20, int(len(lines) * max_tokens / max(chunk.token_estimate, 1)))
    chunks: list[Chunk] = []
    cursor = 0
    while cursor < len(lines):
        part_start = chunk.start_line + cursor
        part_end_idx = min(len(lines), cursor + approx_lines)
        part_end = chunk.start_line + part_end_idx - 1
        part_lines = lines[cursor:part_end_idx]
        text = "\n".join(part_lines)
        content_hash = sha256_text(text)
        chunk_id = sha256_text(f"{chunk.repo_path}:{chunk.file_path}:{part_start}:{part_end}:{content_hash}")[:24]
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                repo_path=chunk.repo_path,
                file_path=chunk.file_path,
                start_line=part_start,
                end_line=part_end,
                language=chunk.language,
                symbol_name=chunk.symbol_name,
                chunk_type=chunk.chunk_type,
                text=text,
                token_estimate=estimate_tokens(text),
                content_hash=content_hash,
            )
        )
        if part_end_idx >= len(lines):
            break
        cursor = max(part_end_idx - overlap_lines, cursor + 1)
    return chunks


def _python_chunks(repo_path: str, file_path: str, text: str, max_tokens: int) -> list[Chunk]:
    lines = text.splitlines()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    chunks: list[Chunk] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if not start or not end:
            continue
        decorators = getattr(node, "decorator_list", [])
        if decorators:
            start = min(start, *(getattr(d, "lineno", start) for d in decorators))
        chunk_type = "class" if isinstance(node, ast.ClassDef) else "function"
        if node.name.startswith("test_") or file_path.split("/")[-1].startswith("test_"):
            chunk_type = "test"
        chunk = make_chunk(repo_path, file_path, "python", lines, start, end, node.name, chunk_type)
        chunks.extend(_split_long_chunk(chunk, max_tokens, DEFAULT_CHUNK_OVERLAP_LINES))
    return _dedupe_chunks(chunks)


C_LIKE_SYMBOL_RE = re.compile(
    r"^\s*(?:(?:export|public|private|protected|static|async|final|fn|func|def|class|interface|struct|enum|type)\s+)*"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?:[({:<]|=|extends|implements)"
)


def _brace_chunks(repo_path: str, file_path: str, language: str, text: str, max_tokens: int) -> list[Chunk]:
    lines = text.splitlines()
    chunks: list[Chunk] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        match = C_LIKE_SYMBOL_RE.match(line)
        if not match or "{" not in line and language not in {"rust", "go"}:
            i += 1
            continue
        start = i + 1
        name = match.group("name")
        brace_depth = 0
        seen_open = False
        j = i
        while j < len(lines):
            brace_depth += lines[j].count("{")
            if lines[j].count("{"):
                seen_open = True
            brace_depth -= lines[j].count("}")
            if seen_open and brace_depth <= 0:
                break
            j += 1
        end = min(j + 1, len(lines))
        ctype = "class" if re.search(r"\b(class|interface|struct|enum)\b", line) else "function"
        if "test" in name.lower() or "/test" in file_path.lower():
            ctype = "test"
        chunk = make_chunk(repo_path, file_path, language, lines, start, end, name, ctype)
        if chunk.token_estimate >= 10:
            chunks.extend(_split_long_chunk(chunk, max_tokens, DEFAULT_CHUNK_OVERLAP_LINES))
        i = max(j + 1, i + 1)
    return _dedupe_chunks(chunks)


def _markdown_chunks(repo_path: str, file_path: str, language: str, text: str, max_tokens: int) -> list[Chunk]:
    lines = text.splitlines()
    starts = [idx + 1 for idx, line in enumerate(lines) if line.startswith("#")]
    if not starts:
        return _line_window_chunks(repo_path, file_path, language, text, max_tokens=max_tokens, chunk_type="docs")
    starts.append(len(lines) + 1)
    chunks: list[Chunk] = []
    for current, nxt in zip(starts, starts[1:]):
        end = nxt - 1
        heading = lines[current - 1].lstrip("#").strip() or None
        chunk = make_chunk(repo_path, file_path, language, lines, current, end, heading, "docs")
        chunks.extend(_split_long_chunk(chunk, max_tokens, DEFAULT_CHUNK_OVERLAP_LINES))
    return chunks


def _config_chunks(repo_path: str, file_path: str, language: str, text: str, max_tokens: int) -> list[Chunk]:
    return _line_window_chunks(repo_path, file_path, language, text, max_tokens=max_tokens, chunk_type="config", window_lines=80, overlap=10)


def _line_window_chunks(
    repo_path: str,
    file_path: str,
    language: str,
    text: str,
    max_tokens: int,
    chunk_type: str = "unknown",
    window_lines: int = DEFAULT_LINE_WINDOW,
    overlap: int = DEFAULT_CHUNK_OVERLAP_LINES,
) -> list[Chunk]:
    lines = text.splitlines()
    if not lines:
        return []
    chunks: list[Chunk] = []
    cursor = 1
    while cursor <= len(lines):
        end = min(len(lines), cursor + window_lines - 1)
        chunk = make_chunk(repo_path, file_path, language, lines, cursor, end, None, chunk_type)
        if chunk.token_estimate > max_tokens and end > cursor:
            chunks.extend(_split_long_chunk(chunk, max_tokens, overlap))
        else:
            chunks.append(chunk)
        if end == len(lines):
            break
        cursor = max(cursor + 1, end - overlap + 1)
    return chunks


def _dedupe_chunks(chunks: list[Chunk]) -> list[Chunk]:
    seen: set[tuple[int, int, str | None]] = set()
    out: list[Chunk] = []
    for chunk in sorted(chunks, key=lambda c: (c.start_line, c.end_line, c.symbol_name or "")):
        key = (chunk.start_line, chunk.end_line, chunk.symbol_name)
        if key in seen:
            continue
        seen.add(key)
        out.append(chunk)
    return out


def chunk_text(
    repo_path: str,
    file_path: str,
    text: str,
    language: str | None = None,
    max_tokens: int = DEFAULT_CHUNK_TOKEN_LIMIT,
) -> list[Chunk]:
    language = language or language_for_path(file_path) or "unknown"
    if language == "python":
        chunks = _python_chunks(repo_path, file_path, text, max_tokens=max_tokens)
        if chunks:
            return chunks
    if language in {"javascript", "typescript", "java", "go", "rust", "c", "cpp"}:
        chunks = _brace_chunks(repo_path, file_path, language, text, max_tokens=max_tokens)
        if chunks:
            return chunks
    if language == "markdown":
        return _markdown_chunks(repo_path, file_path, language, text, max_tokens=max_tokens)
    if language in {"yaml", "json", "toml", "config"}:
        return _config_chunks(repo_path, file_path, language, text, max_tokens=max_tokens)
    return _line_window_chunks(repo_path, file_path, language, text, max_tokens=max_tokens)


def chunk_file(repo_path: str, root: Path, rel_path: str, max_tokens: int = DEFAULT_CHUNK_TOKEN_LIMIT) -> list[Chunk]:
    text = (root / rel_path).read_text(encoding="utf-8")
    return chunk_text(repo_path, rel_path, text, language_for_path(rel_path), max_tokens=max_tokens)
