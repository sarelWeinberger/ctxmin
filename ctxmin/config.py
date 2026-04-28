from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_NAME = "google/embeddinggemma-300m"
DEFAULT_TOKEN_BUDGET = 6000
DEFAULT_CHUNK_TOKEN_LIMIT = 700
DEFAULT_CHUNK_OVERLAP_LINES = 20
DEFAULT_LINE_WINDOW = 120
DEFAULT_EMBED_BATCH_SIZE = 16
CHUNKER_VERSION = "2"
NORMALIZATION_VERSION = "1"

INDEX_ROOT = Path.home() / ".ctxmin" / "indexes"
CONTEXTBENCH_ROOT = Path.home() / ".ctxmin" / "contextbench"

SKIP_DIRS = {
    ".git",
    "node_modules",
    "dist",
    "build",
    "target",
    ".venv",
    "venv",
    "__pycache__",
    "coverage",
    ".next",
    ".turbo",
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
}

LOCK_FILE_NAMES = {
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "poetry.lock",
    "Pipfile.lock",
    "Cargo.lock",
    "go.sum",
    "Gemfile.lock",
}

SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".md": "markdown",
    ".markdown": "markdown",
    ".rst": "markdown",
    ".txt": "markdown",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".json": "json",
    ".toml": "toml",
}

BINARY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".bz2",
    ".xz",
    ".7z",
    ".jar",
    ".class",
    ".so",
    ".dylib",
    ".dll",
    ".exe",
    ".bin",
    ".wasm",
    ".mp4",
    ".mov",
    ".mp3",
}


@dataclass(frozen=True)
class RankingWeights:
    dense_similarity: float = 0.26
    lexical_score: float = 0.24
    exact_term_overlap: float = 0.12
    symbol_overlap: float = 0.14
    path_relevance: float = 0.11
    filename_relevance: float = 0.08
    test_file_relevance: float = 0.04
    import_dependency_relevance: float = 0.04
    recency_relevance: float = 0.03
    config_relevance: float = 0.03
    redundancy_penalty: float = 0.12
    low_signal_penalty: float = 0.10
    generated_vendor_penalty: float = 0.45


@dataclass(frozen=True)
class RetrievalConfig:
    dense_top_k: int = 200
    lexical_top_k: int = 200
    max_candidates: int = 300
    rerank_top_k: int = 80
    mmr_top_k: int = 80
    mmr_lambda: float = 0.35
    min_score: float = 0.0
    use_dense: bool = True
    use_lexical: bool = True
    use_reranker: bool = True
    use_mmr: bool = True
    graph_boost: bool = False
    graph_expand_budget: int = 1000
    debug_ranking: bool = False
    max_ram_mb: int | None = None


@dataclass(frozen=True)
class IndexConfig:
    model_name: str = DEFAULT_MODEL_NAME
    token_budget: int = DEFAULT_TOKEN_BUDGET
    chunk_token_limit: int = DEFAULT_CHUNK_TOKEN_LIMIT
    chunk_overlap_lines: int = DEFAULT_CHUNK_OVERLAP_LINES
    line_window: int = DEFAULT_LINE_WINDOW
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE
    include_locks: bool = False
