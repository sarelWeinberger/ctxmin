from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np

from .chunking import Chunk
from .config import INDEX_ROOT
from .utils import normalize_repo_path, repo_hash


def index_path_for_repo(repo_path: str | Path) -> Path:
    return INDEX_ROOT / repo_hash(repo_path)


class IndexStorage:
    def __init__(self, repo_path: str | Path, index_root: Path | None = None) -> None:
        self.repo_path = normalize_repo_path(repo_path)
        self.index_path = (index_root or INDEX_ROOT) / repo_hash(self.repo_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.index_path / "index.sqlite"
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS settings (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS files (
              file_path TEXT PRIMARY KEY,
              file_hash TEXT NOT NULL,
              language TEXT NOT NULL,
              indexed_at REAL NOT NULL,
              skipped_reason TEXT
            );
            CREATE TABLE IF NOT EXISTS skipped_files (
              file_path TEXT PRIMARY KEY,
              reason TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
              chunk_id TEXT PRIMARY KEY,
              file_path TEXT NOT NULL,
              start_line INTEGER NOT NULL,
              end_line INTEGER NOT NULL,
              language TEXT NOT NULL,
              symbol_name TEXT,
              chunk_type TEXT NOT NULL,
              text TEXT NOT NULL,
              token_estimate INTEGER NOT NULL,
              content_hash TEXT NOT NULL,
              cache_key TEXT,
              embedding BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS embedding_cache (
              cache_key TEXT PRIMARY KEY,
              model_name TEXT NOT NULL,
              model_revision TEXT,
              embedding_backend TEXT NOT NULL,
              embedding_dim INTEGER NOT NULL,
              chunk_hash TEXT NOT NULL,
              chunker_version TEXT NOT NULL,
              normalization_version TEXT NOT NULL,
              file_path TEXT NOT NULL,
              file_hash TEXT,
              embedding BLOB NOT NULL,
              created_at REAL NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
              chunk_id UNINDEXED,
              file_path,
              symbol_name,
              chunk_type,
              language,
              text,
              tokenize='unicode61 tokenchars ''_./-'''
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);
            CREATE INDEX IF NOT EXISTS idx_chunks_symbol ON chunks(symbol_name);
            """
        )
        columns = {row["name"] for row in self.conn.execute("PRAGMA table_info(chunks)")}
        if "cache_key" not in columns:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN cache_key TEXT")
        self.conn.commit()

    def set_setting(self, key: str, value: object) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO settings(key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
        self.conn.commit()

    def get_setting(self, key: str, default: object | None = None) -> object | None:
        row = self.conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        if not row:
            return default
        return json.loads(row["value"])

    def file_hash(self, file_path: str) -> str | None:
        row = self.conn.execute("SELECT file_hash FROM files WHERE file_path = ?", (file_path,)).fetchone()
        return row["file_hash"] if row else None

    def count_chunks_for_file(self, file_path: str) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM chunks WHERE file_path = ?", (file_path,)).fetchone()
        return int(row["n"] if row else 0)

    def count_chunks(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
        return int(row["n"] if row else 0)

    def mark_file(self, file_path: str, file_hash: str, language: str, indexed_at: float) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO files(file_path, file_hash, language, indexed_at, skipped_reason)
            VALUES (?, ?, ?, ?, NULL)
            """,
            (file_path, file_hash, language, indexed_at),
        )
        self.conn.execute("DELETE FROM skipped_files WHERE file_path = ?", (file_path,))

    def mark_skipped(self, file_path: str, reason: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO skipped_files(file_path, reason) VALUES (?, ?)",
            (file_path, reason),
        )

    def clear_skipped(self, file_path: str) -> None:
        self.conn.execute("DELETE FROM skipped_files WHERE file_path = ?", (file_path,))

    def delete_file(self, file_path: str) -> None:
        self.conn.execute(
            "DELETE FROM chunk_fts WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE file_path = ?)",
            (file_path,),
        )
        self.conn.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))
        self.conn.execute("DELETE FROM files WHERE file_path = ?", (file_path,))
        self.conn.execute("DELETE FROM skipped_files WHERE file_path = ?", (file_path,))

    def remove_files_not_in(self, file_paths: set[str]) -> None:
        existing = {row["file_path"] for row in self.conn.execute("SELECT file_path FROM files")}
        for deleted in existing - file_paths:
            self.delete_file(deleted)
        self.conn.commit()

    def get_cached_embedding(self, cache_key: str, embedding_dim: int) -> np.ndarray | None:
        row = self.conn.execute(
            "SELECT embedding, embedding_dim FROM embedding_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if not row or int(row["embedding_dim"]) != int(embedding_dim):
            return None
        embedding = np.frombuffer(row["embedding"], dtype=np.float32)
        if embedding.shape[0] != embedding_dim:
            return None
        return embedding.copy()

    def store_cached_embedding(
        self,
        cache_key: str,
        embedding: np.ndarray,
        *,
        model_name: str,
        model_revision: str | None,
        embedding_backend: str,
        embedding_dim: int,
        chunk_hash: str,
        chunker_version: str,
        normalization_version: str,
        file_path: str,
        file_hash: str | None,
        created_at: float,
    ) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO embedding_cache(
              cache_key, model_name, model_revision, embedding_backend, embedding_dim,
              chunk_hash, chunker_version, normalization_version, file_path, file_hash,
              embedding, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cache_key,
                model_name,
                model_revision,
                embedding_backend,
                int(embedding_dim),
                chunk_hash,
                chunker_version,
                normalization_version,
                file_path,
                file_hash,
                np.asarray(embedding, dtype=np.float32).tobytes(),
                created_at,
            ),
        )

    def replace_chunks_for_file(self, file_path: str, chunks: Iterable[Chunk], embeddings: np.ndarray) -> None:
        self.conn.execute("DELETE FROM chunk_fts WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE file_path = ?)", (file_path,))
        self.conn.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))
        rows = []
        fts_rows = []
        for chunk, embedding in zip(chunks, embeddings):
            rows.append(
                (
                    chunk.chunk_id,
                    chunk.file_path,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.language,
                    chunk.symbol_name,
                    chunk.chunk_type,
                    chunk.text,
                    chunk.token_estimate,
                    chunk.content_hash,
                    chunk.cache_key,
                    np.asarray(embedding, dtype=np.float32).tobytes(),
                )
            )
            fts_rows.append(
                (
                    chunk.chunk_id,
                    chunk.file_path,
                    chunk.symbol_name or "",
                    chunk.chunk_type,
                    chunk.language,
                    chunk.text,
                )
            )
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO chunks(
              chunk_id, file_path, start_line, end_line, language, symbol_name, chunk_type,
              text, token_estimate, content_hash, cache_key, embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.executemany(
            """
            INSERT INTO chunk_fts(chunk_id, file_path, symbol_name, chunk_type, language, text)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            fts_rows,
        )

    def commit(self) -> None:
        self.conn.commit()

    def load_chunks(self) -> list[Chunk]:
        dim = int(self.get_setting("embedding_dimension", 0) or 0)
        chunks: list[Chunk] = []
        for row in self.conn.execute(
            """
            SELECT chunk_id, file_path, start_line, end_line, language, symbol_name, chunk_type,
                   text, token_estimate, content_hash, cache_key, embedding
            FROM chunks
            ORDER BY file_path, start_line, end_line
            """
        ):
            embedding = np.frombuffer(row["embedding"], dtype=np.float32)
            if dim and embedding.shape[0] != dim:
                embedding = embedding[:dim]
            chunks.append(
                Chunk(
                    chunk_id=row["chunk_id"],
                    repo_path=str(self.repo_path),
                    file_path=row["file_path"],
                    start_line=row["start_line"],
                    end_line=row["end_line"],
                    language=row["language"],
                    symbol_name=row["symbol_name"],
                    chunk_type=row["chunk_type"],
                    text=row["text"],
                    token_estimate=row["token_estimate"],
                    content_hash=row["content_hash"],
                    embedding=embedding.copy(),
                    cache_key=row["cache_key"],
                )
            )
        return chunks

    def load_chunks_by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        dim = int(self.get_setting("embedding_dimension", 0) or 0)
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = self.conn.execute(
            f"""
            SELECT chunk_id, file_path, start_line, end_line, language, symbol_name, chunk_type,
                   text, token_estimate, content_hash, cache_key, embedding
            FROM chunks
            WHERE chunk_id IN ({placeholders})
            """,
            chunk_ids,
        ).fetchall()
        by_id = {}
        for row in rows:
            embedding = np.frombuffer(row["embedding"], dtype=np.float32)
            if dim and embedding.shape[0] != dim:
                embedding = embedding[:dim]
            by_id[row["chunk_id"]] = Chunk(
                chunk_id=row["chunk_id"],
                repo_path=str(self.repo_path),
                file_path=row["file_path"],
                start_line=row["start_line"],
                end_line=row["end_line"],
                language=row["language"],
                symbol_name=row["symbol_name"],
                chunk_type=row["chunk_type"],
                text=row["text"],
                token_estimate=row["token_estimate"],
                content_hash=row["content_hash"],
                embedding=embedding.copy(),
                cache_key=row["cache_key"],
            )
        return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]

    def iter_chunk_embeddings(self):
        dim = int(self.get_setting("embedding_dimension", 0) or 0)
        for row in self.conn.execute(
            """
            SELECT chunk_id, file_path, start_line, end_line, language, symbol_name, chunk_type,
                   token_estimate, content_hash, cache_key, embedding
            FROM chunks
            ORDER BY chunk_id
            """
        ):
            embedding = np.frombuffer(row["embedding"], dtype=np.float32)
            if dim and embedding.shape[0] != dim:
                embedding = embedding[:dim]
            yield row, embedding

    def search_fts(self, query: str, limit: int) -> list[tuple[str, float]]:
        try:
            rows = self.conn.execute(
                """
                SELECT chunk_id, bm25(chunk_fts, 2.0, 1.5, 1.0, 0.5, 1.0) AS rank
                FROM chunk_fts
                WHERE chunk_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()
            if not rows:
                return []
            total = max(1, len(rows))
            return [(row["chunk_id"], 1.0 - (idx / total)) for idx, row in enumerate(rows)]
        except sqlite3.OperationalError:
            return []

    def stats(self) -> dict[str, object]:
        files = self.conn.execute("SELECT COUNT(*) AS n FROM files").fetchone()["n"]
        chunks = self.conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        skipped = self.conn.execute("SELECT COUNT(*) AS n FROM skipped_files").fetchone()["n"]
        languages = {
            row["language"]: row["n"]
            for row in self.conn.execute("SELECT language, COUNT(*) AS n FROM files GROUP BY language ORDER BY n DESC")
        }
        return {
            "repo_path": str(self.repo_path),
            "storage_location": str(self.index_path),
            "files_indexed": files,
            "chunks": chunks,
            "skipped_files": skipped,
            "languages": languages,
            "embedding_model": self.get_setting("embedding_model", "unknown"),
            "embedding_backend": self.get_setting("embedding_backend", "unknown"),
            "embedding_dimension": self.get_setting("embedding_dimension", "unknown"),
        }

    def skipped_files(self, limit: int = 25) -> list[dict[str, str]]:
        rows = self.conn.execute(
            "SELECT file_path, reason FROM skipped_files ORDER BY file_path LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
