from __future__ import annotations

import time
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import chunk_text
from .config import CHUNKER_VERSION, NORMALIZATION_VERSION, IndexConfig
from .models import EmbeddingModel, HashEmbeddingModel
from .profiling import Profile
from .storage import IndexStorage
from .utils import (
    git_tracked_files,
    has_skipped_part,
    is_generated_text,
    is_git_repo,
    is_lock_file,
    is_probably_binary,
    language_for_path,
    normalize_repo_path,
    safe_read_text,
    sha256_bytes,
    sha256_text,
    walk_supported_files,
)


@dataclass
class IndexReport:
    repo_path: str
    storage_location: str
    files_seen: int = 0
    files_indexed: int = 0
    files_unchanged: int = 0
    files_skipped: int = 0
    chunks_indexed: int = 0
    chunks_reused: int = 0
    chunks_reembedded: int = 0
    chunks_invalidated: int = 0
    skipped_reasons: dict[str, int] = field(default_factory=dict)
    embedding_model: str = ""
    embedding_backend: str = ""
    embedding_dimension: int = 0


class RepoIndexer:
    def __init__(
        self,
        repo_path: str | Path,
        config: IndexConfig | None = None,
        model: EmbeddingModel | HashEmbeddingModel | None = None,
        profile: Profile | None = None,
    ) -> None:
        self.repo_path = normalize_repo_path(repo_path)
        self.config = config or IndexConfig()
        self.model = model
        self.profile = profile or Profile(enabled=False)
        self.storage = IndexStorage(self.repo_path)

    def close(self) -> None:
        self.storage.close()

    def _candidate_files(self) -> list[str]:
        tracked = git_tracked_files(self.repo_path) if is_git_repo(self.repo_path) else []
        return tracked or walk_supported_files(self.repo_path)

    def _skip_reason(self, rel_path: str) -> str | None:
        full_path = self.repo_path / rel_path
        language = language_for_path(rel_path)
        if has_skipped_part(rel_path):
            return "skip_dir"
        if not language:
            return "unsupported_extension"
        if is_lock_file(rel_path) and not self.config.include_locks:
            return "lock_file"
        if not full_path.exists() or not full_path.is_file():
            return "missing_or_not_file"
        if full_path.stat().st_size > 1_500_000:
            return "too_large"
        if is_probably_binary(full_path):
            return "binary"
        return None

    def _cache_key(
        self,
        *,
        file_path: str,
        chunk_hash: str,
        model_name: str,
        model_revision: str | None,
        embedding_backend: str,
        embedding_dim: int,
    ) -> str:
        return sha256_text(
            "\n".join(
                [
                    f"model={model_name}",
                    f"revision={model_revision or 'default'}",
                    f"backend={embedding_backend}",
                    f"dim={embedding_dim}",
                    f"chunker={CHUNKER_VERSION}",
                    f"normalization={NORMALIZATION_VERSION}",
                    f"path={file_path}",
                    f"chunk={chunk_hash}",
                ]
            )
        )

    def build_or_update(self) -> IndexReport:
        if self.model is None:
            self.model = EmbeddingModel.load(
                model_name=self.config.model_name,
                batch_size=self.config.embed_batch_size,
            )
        report = IndexReport(
            repo_path=str(self.repo_path),
            storage_location=str(self.storage.index_path),
            embedding_model=getattr(self.model, "model_name", self.config.model_name),
            embedding_backend=getattr(self.model, "backend", "unknown"),
        )
        embedding_dimension = int(getattr(self.model, "dimension", 0) or 0)
        model_revision = getattr(self.model, "model_revision", None)
        report.embedding_dimension = embedding_dimension
        previous_model = self.storage.get_setting("embedding_model")
        previous_backend = self.storage.get_setting("embedding_backend")
        previous_dimension = self.storage.get_setting("embedding_dimension")
        previous_chunker = self.storage.get_setting("chunker_version")
        previous_normalization = self.storage.get_setting("normalization_version")
        force_reembed = bool(
            previous_model
            and (
                previous_model != report.embedding_model
                or previous_backend != report.embedding_backend
                or int(previous_dimension or 0) != embedding_dimension
                or previous_chunker != CHUNKER_VERSION
                or previous_normalization != NORMALIZATION_VERSION
            )
        )
        if force_reembed:
            report.chunks_invalidated = len(self.storage.load_chunks())
        self.storage.set_setting("repo_path", str(self.repo_path))
        self.storage.set_setting("embedding_model", report.embedding_model)
        self.storage.set_setting("embedding_backend", report.embedding_backend)
        self.storage.set_setting("embedding_dimension", embedding_dimension)
        self.storage.set_setting("model_revision", model_revision or "default")
        self.storage.set_setting("chunker_version", CHUNKER_VERSION)
        self.storage.set_setting("normalization_version", NORMALIZATION_VERSION)

        with self.profile.timer("scan_repo_time"):
            candidates = self._candidate_files()
        report.files_seen = len(candidates)
        seen_indexable: set[str] = set()
        for rel_path in candidates:
            reason = self._skip_reason(rel_path)
            if reason:
                report.files_skipped += 1
                report.skipped_reasons[reason] = report.skipped_reasons.get(reason, 0) + 1
                self.storage.mark_skipped(rel_path, reason)
                continue
            full_path = self.repo_path / rel_path
            with self.profile.timer("hash_files_time"):
                data = full_path.read_bytes()
                file_hash = sha256_bytes(data)
            seen_indexable.add(rel_path)
            if not force_reembed and self.storage.file_hash(rel_path) == file_hash:
                self.storage.clear_skipped(rel_path)
                report.files_unchanged += 1
                existing_count = self.storage.count_chunks_for_file(rel_path)
                report.chunks_reused += existing_count
                self.profile.incr("number_of_chunks_loaded_from_cache", existing_count)
                continue
            text = safe_read_text(full_path)
            if text is None:
                report.files_skipped += 1
                report.skipped_reasons["unreadable"] = report.skipped_reasons.get("unreadable", 0) + 1
                self.storage.mark_skipped(rel_path, "unreadable")
                continue
            if is_generated_text(text, rel_path):
                report.files_skipped += 1
                report.skipped_reasons["generated"] = report.skipped_reasons.get("generated", 0) + 1
                self.storage.mark_skipped(rel_path, "generated")
                continue
            language = language_for_path(rel_path) or "unknown"
            chunks = chunk_text(
                str(self.repo_path),
                rel_path,
                text,
                language=language,
                max_tokens=self.config.chunk_token_limit,
            )
            if not chunks:
                self.storage.delete_file(rel_path)
                continue
            embeddings_by_index: list[object | None] = [None] * len(chunks)
            missing_indexes: list[int] = []
            for idx, chunk in enumerate(chunks):
                cache_key = self._cache_key(
                    file_path=chunk.file_path,
                    chunk_hash=chunk.content_hash,
                    model_name=report.embedding_model,
                    model_revision=model_revision,
                    embedding_backend=report.embedding_backend,
                    embedding_dim=embedding_dimension,
                )
                chunk.cache_key = cache_key
                cached = None if force_reembed else self.storage.get_cached_embedding(cache_key, embedding_dimension)
                if cached is None:
                    missing_indexes.append(idx)
                else:
                    embeddings_by_index[idx] = cached
                    report.chunks_reused += 1
                    self.profile.incr("number_of_chunks_loaded_from_cache")
            if missing_indexes:
                with self.profile.timer("embed_changed_chunks_time"):
                    new_embeddings = self.model.embed_documents([chunks[idx].text for idx in missing_indexes])
                now = time.time()
                for local_idx, embedding in zip(missing_indexes, new_embeddings):
                    chunk = chunks[local_idx]
                    embeddings_by_index[local_idx] = embedding
                    self.storage.store_cached_embedding(
                        chunk.cache_key or "",
                        embedding,
                        model_name=report.embedding_model,
                        model_revision=model_revision,
                        embedding_backend=report.embedding_backend,
                        embedding_dim=embedding_dimension,
                        chunk_hash=chunk.content_hash,
                        chunker_version=CHUNKER_VERSION,
                        normalization_version=NORMALIZATION_VERSION,
                        file_path=chunk.file_path,
                        file_hash=file_hash,
                        created_at=now,
                    )
                report.chunks_reembedded += len(missing_indexes)
                self.profile.incr("number_of_chunks_embedded", len(missing_indexes))
            embeddings = [embedding for embedding in embeddings_by_index if embedding is not None]
            self.storage.replace_chunks_for_file(rel_path, chunks, embeddings)
            self.storage.mark_file(rel_path, file_hash, language, time.time())
            self.storage.commit()
            report.files_indexed += 1
            report.chunks_indexed += len(chunks)
        self.storage.remove_files_not_in(seen_indexable)
        self.profile.incr("number_of_chunks_invalidated", report.chunks_invalidated)
        if self.profile.enabled:
            print(
                "Embedding cache:\n"
                f"- reused: {report.chunks_reused} chunks\n"
                f"- re-embedded: {report.chunks_reembedded} chunks\n"
                f"- invalidated: {report.chunks_invalidated} chunks\n"
                f"- backend: {report.embedding_backend}\n"
                f"- model: {report.embedding_model}\n"
                f"- embedding_dim: {report.embedding_dimension}",
                file=sys.stderr,
            )
        return report
