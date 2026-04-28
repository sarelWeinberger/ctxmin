from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .config import DEFAULT_EMBED_BATCH_SIZE, DEFAULT_MODEL_NAME


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors.astype(np.float32, copy=False)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class HashEmbeddingModel:
    """Deterministic local fallback used when sentence-transformers is unavailable."""

    def __init__(self, dimension: int = 384, model_name: str = "local-hash-fallback") -> None:
        self.dimension = dimension
        self.model_name = model_name
        self.model_revision = "hash"
        self.device = "cpu"
        self.backend = "hash"

    def _embed_one(self, text: str) -> np.ndarray:
        import hashlib
        import re

        vec = np.zeros(self.dimension, dtype=np.float32)
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_./-]*", text.lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm:
            vec /= norm
        return vec

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_one(text)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack([self._embed_one(text) for text in texts]).astype(np.float32)


@dataclass
class EmbeddingModel:
    model_name: str = DEFAULT_MODEL_NAME
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE
    device: str | None = None
    dtype: str = "float32"

    def __post_init__(self) -> None:
        self.backend = "sentence-transformers"
        self.model_revision = os.environ.get("CTXMIN_MODEL_REVISION") or "default"
        self.model = self._load_model()
        self.dimension = self._detect_dimension()

    @classmethod
    def load(
        cls,
        model_name: str = DEFAULT_MODEL_NAME,
        batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        allow_fallback: bool = True,
    ) -> "EmbeddingModel | HashEmbeddingModel":
        requested = os.environ.get("CTXMIN_EMBEDDING_BACKEND", "").strip().lower()
        if requested == "hash":
            return HashEmbeddingModel()
        try:
            return cls(model_name=model_name, batch_size=batch_size)
        except Exception as exc:
            if not allow_fallback:
                raise
            warnings.warn(
                f"Could not load {model_name!r} with sentence-transformers ({exc}). "
                "Using deterministic local hash fallback.",
                RuntimeWarning,
            )
            return HashEmbeddingModel(model_name=f"{model_name}+hash-fallback")

    def _choose_device(self) -> str:
        if self.device:
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _load_model(self):
        from sentence_transformers import SentenceTransformer

        device = self._choose_device()
        kwargs = {"device": device}
        if self.dtype == "bfloat16":
            try:
                import torch

                kwargs["model_kwargs"] = {"torch_dtype": torch.bfloat16}
            except Exception:
                pass
        # float16 is intentionally not used; CPU defaults to float32.
        self.device = device
        return SentenceTransformer(self.model_name, **kwargs)

    def _detect_dimension(self) -> int:
        if hasattr(self.model, "get_sentence_embedding_dimension"):
            dim = self.model.get_sentence_embedding_dimension()
            if dim:
                return int(dim)
        sample = self.embed_query("ctxmin dimension probe")
        return int(sample.shape[0])

    def _encode(self, texts: list[str], method_names: Iterable[str]) -> np.ndarray:
        method = None
        for name in method_names:
            candidate = getattr(self.model, name, None)
            if callable(candidate):
                method = candidate
                break
        if method is None:
            method = self.model.encode
        try:
            vectors = method(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except TypeError:
            if method is not self.model.encode:
                method = self.model.encode
            vectors = method(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return _normalize(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([text], ("encode_query", "encode"))[0]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return self._encode(texts, ("encode_document", "encode"))
