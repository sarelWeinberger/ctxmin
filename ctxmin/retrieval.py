from __future__ import annotations

import heapq
import math
import re
from pathlib import Path

import numpy as np

from .chunking import Chunk
from .config import RankingWeights, RetrievalConfig
from .models import EmbeddingModel, HashEmbeddingModel
from .profiling import Profile
from .prompt_analyzer import PromptAnalysis
from .ranking import RankedChunk, ScoreComponents, rank_chunks
from .storage import IndexStorage
from .utils import common_path_suffix_match, git_changed_files, git_recent_files, tokenize


IMPORT_RE = re.compile(
    r"^\s*(?:from\s+([A-Za-z0-9_.]+)\s+import|import\s+([A-Za-z0-9_.,\s]+)|"
    r"(?:import|require)\(['\"]([^'\"]+)['\"]\))",
    re.MULTILINE,
)
FTS_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./-]{1,}|[A-Z][A-Za-z0-9_]+")
PENALIZED_PATH_PARTS = {
    "vendor",
    "node_modules",
    "dist",
    "build",
    "generated",
    ".cache",
    ".venv",
    "venv",
    "__pycache__",
}


def _normalize_score(raw: float) -> float:
    return max(0.0, min(1.0, (raw + 1.0) / 2.0))


def _lexical_query(analysis: PromptAnalysis, max_terms: int = 32) -> str:
    terms: list[str] = []
    seed = " ".join(
        [
            analysis.query_text(),
            " ".join(analysis.symbols),
            " ".join(analysis.explicit_file_paths),
            " ".join(analysis.failing_tests),
            " ".join(analysis.commands),
        ]
    )
    for term in FTS_TOKEN_RE.findall(seed):
        cleaned = term.strip("./-").lower()
        if len(cleaned) < 3:
            continue
        if cleaned not in terms:
            terms.append(cleaned)
        if len(terms) >= max_terms:
            break
    return " OR ".join(f'"{term}"' for term in terms)


def _dense_search(storage: IndexStorage, query_vec: np.ndarray, top_k: int) -> dict[str, float]:
    if top_k <= 0:
        return {}
    q = np.asarray(query_vec, dtype=np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm:
        q = q / q_norm
    heap: list[tuple[float, str]] = []
    for row, embedding in storage.iter_chunk_embeddings():
        score = float(np.asarray(embedding, dtype=np.float32) @ q)
        item = (score, row["chunk_id"])
        if len(heap) < top_k:
            heapq.heappush(heap, item)
        elif score > heap[0][0]:
            heapq.heapreplace(heap, item)
    return {chunk_id: _normalize_score(score) for score, chunk_id in heap}


def _term_overlap(prompt_tokens: set[str], text: str) -> float:
    if not prompt_tokens:
        return 0.0
    chunk_tokens = tokenize(text)
    if not chunk_tokens:
        return 0.0
    overlap = len(prompt_tokens & chunk_tokens)
    return min(1.0, overlap / max(4.0, math.sqrt(len(prompt_tokens))))


def _symbol_overlap(analysis: PromptAnalysis, chunk: Chunk) -> float:
    score = 0.0
    for symbol in analysis.symbols:
        if chunk.symbol_name and symbol.lower() == chunk.symbol_name.lower():
            score = max(score, 1.0)
        elif re.search(rf"\b{re.escape(symbol)}\b", chunk.text):
            score = max(score, 0.75)
    return score


def path_relevance(analysis: PromptAnalysis, file_path: str) -> float:
    lower_prompt = analysis.prompt.lower()
    lower_path = file_path.lower()
    score = 0.0
    for path in analysis.explicit_file_paths:
        if path == file_path or common_path_suffix_match(path, file_path):
            score = max(score, 1.0)
        elif Path(path).stem and Path(path).stem.lower() in lower_path:
            score = max(score, 0.7)
    categories = [
        ({"benchmark", "bench", "contextbench", "eval", "evaluation"}, ["bench", "benchmark", "eval", "evaluation", "contextbench", "tests"]),
        ({"cli", "command", "commands"}, ["cli", "commands", "scripts", "__main__"]),
        ({"config", "setting", "settings"}, ["config", "yaml", "toml", "json", "settings"]),
        ({"dataset", "data", "fixture", "fixtures"}, ["dataset", "loader", "data", "fixtures"]),
    ]
    for prompt_words, path_words in categories:
        if any(word in lower_prompt for word in prompt_words) and any(word in lower_path for word in path_words):
            score = max(score, 0.65)
    return score


def filename_relevance(analysis: PromptAnalysis, file_path: str) -> float:
    prompt_tokens = tokenize(analysis.query_text())
    stem = Path(file_path).stem.lower()
    if not stem:
        return 0.0
    if stem in prompt_tokens:
        return 1.0
    parts = {part for part in re.split(r"[_\-.]+", stem) if len(part) > 2}
    if not parts:
        return 0.0
    return min(1.0, len(parts & prompt_tokens) / len(parts))


def generated_vendor_penalty(file_path: str) -> float:
    lower = file_path.lower()
    parts = set(Path(lower).parts)
    if parts & PENALIZED_PATH_PARTS:
        return 1.0
    if lower.endswith((".min.js", ".bundle.js")) or "lock" in Path(lower).name:
        return 0.8
    return 0.0


def low_signal_penalty(chunk: Chunk) -> float:
    stripped = chunk.text.strip()
    if chunk.token_estimate < 8:
        return 0.7
    if not stripped:
        return 1.0
    alpha = sum(1 for c in stripped if c.isalpha())
    if alpha < max(8, len(stripped) * 0.05):
        return 0.5
    return 0.0


def test_relevance(analysis: PromptAnalysis, chunk: Chunk) -> float:
    lower_path = chunk.file_path.lower()
    lower_prompt = analysis.prompt.lower()
    if chunk.chunk_type == "test" or "/test" in lower_path or lower_path.startswith("test"):
        if analysis.failing_tests or "test" in lower_prompt or "pytest" in lower_prompt:
            return 1.0
        return 0.25
    return 0.0


def config_relevance(analysis: PromptAnalysis, chunk: Chunk) -> float:
    lower = analysis.prompt.lower()
    config_words = {"config", "dependency", "build", "lint", "typing", "ci", "workflow", "package", "pyproject"}
    if chunk.chunk_type == "config" and any(word in lower for word in config_words):
        return 1.0
    if Path(chunk.file_path).name.lower() in {"readme.md", "contributing.md"} and any(
        word in lower for word in {"convention", "style", "readme", "docs"}
    ):
        return 0.8
    return 0.0


def _related_files(analysis: PromptAnalysis, chunks: list[Chunk]) -> set[str]:
    mentioned: set[str] = set()
    all_files = {chunk.file_path for chunk in chunks}
    for path in analysis.explicit_file_paths:
        for file_path in all_files:
            if common_path_suffix_match(path, file_path):
                mentioned.add(file_path)
    related: set[str] = set()
    for file_path in mentioned:
        stem = Path(file_path).stem.replace("test_", "")
        for candidate in all_files:
            cstem = Path(candidate).stem.replace("test_", "")
            if stem and stem == cstem and candidate != file_path:
                related.add(candidate)
    return related


def _merge_candidates(
    dense_scores: dict[str, float],
    lexical_scores: dict[str, float],
    max_candidates: int,
) -> list[str]:
    ids = set(dense_scores) | set(lexical_scores)
    scored = [
        (0.55 * dense_scores.get(chunk_id, 0.0) + 0.45 * lexical_scores.get(chunk_id, 0.0), chunk_id)
        for chunk_id in ids
    ]
    scored.sort(reverse=True)
    return [chunk_id for _, chunk_id in scored[:max_candidates]]


class Retriever:
    def __init__(
        self,
        repo_path: str | Path,
        model: EmbeddingModel | HashEmbeddingModel,
        weights: RankingWeights | None = None,
        config: RetrievalConfig | None = None,
        profile: Profile | None = None,
    ) -> None:
        self.repo_path = Path(repo_path).expanduser().resolve()
        self.model = model
        self.weights = weights or RankingWeights()
        self.config = config or RetrievalConfig()
        self.profile = profile or Profile(enabled=False)
        self.storage = IndexStorage(self.repo_path)

    def close(self) -> None:
        self.storage.close()

    def _lexical_search(self, analysis: PromptAnalysis) -> dict[str, float]:
        if not self.config.use_lexical:
            return {}
        query = _lexical_query(analysis)
        if not query:
            return {}
        return dict(self.storage.search_fts(query, self.config.lexical_top_k))

    def _dense_search(self, analysis: PromptAnalysis) -> dict[str, float]:
        if not self.config.use_dense:
            return {}
        query_vec = self.model.embed_query(analysis.query_text())
        return _dense_search(self.storage, query_vec, self.config.dense_top_k)

    def _graph_boost(self, analysis: PromptAnalysis, chunks: list[Chunk], components_by_id: dict[str, ScoreComponents]) -> None:
        if not self.config.graph_boost:
            return
        seed_symbols = [chunk.symbol_name for chunk in chunks[:20] if chunk.symbol_name]
        budget = self.config.graph_expand_budget
        spent = 0
        for symbol in seed_symbols:
            if not symbol or spent >= budget:
                break
            for chunk in chunks:
                if chunk.chunk_id not in components_by_id:
                    continue
                if chunk.symbol_name == symbol:
                    continue
                if re.search(rf"\b{re.escape(symbol)}\b", chunk.text):
                    components_by_id[chunk.chunk_id].import_dependency_relevance = max(
                        components_by_id[chunk.chunk_id].import_dependency_relevance, 0.6
                    )
                    spent += chunk.token_estimate
                    if spent >= budget:
                        break
        self.profile.set("graph_index_mem_mb", 0)

    def retrieve(self, analysis: PromptAnalysis, top_k: int | None = None) -> list[RankedChunk]:
        top_k = top_k or self.config.rerank_top_k
        with self.profile.timer("load_index_time"):
            total_chunks = self.storage.count_chunks()
        self.profile.set("number_of_chunks_total", total_chunks)

        with self.profile.timer("vector_search_time"):
            dense_scores = self._dense_search(analysis)
        self.profile.set("number_of_candidates_dense", len(dense_scores))

        with self.profile.timer("lexical_search_time"):
            lexical_scores = self._lexical_search(analysis)
        self.profile.set("number_of_candidates_lexical", len(lexical_scores))
        self.profile.set("lexical_index_mem_mb", 0)

        with self.profile.timer("candidate_merge_time"):
            merged_ids = _merge_candidates(dense_scores, lexical_scores, self.config.rerank_top_k * 4)
            merged_ids = merged_ids[: max(1, min(len(merged_ids), getattr(self.config, "max_candidates", 300)))]
        self.profile.set("number_of_candidates_after_merge", len(merged_ids))
        self.profile.set("candidate_count", len(merged_ids))

        with self.profile.timer("load_index_time"):
            candidates = self.storage.load_chunks_by_ids(merged_ids)
        by_id = {chunk.chunk_id: chunk for chunk in candidates}
        candidates = [by_id[chunk_id] for chunk_id in merged_ids if chunk_id in by_id]

        prompt_tokens = tokenize(analysis.query_text())
        changed = git_changed_files(self.repo_path)
        recent = git_recent_files(self.repo_path, limit=80)
        related = _related_files(analysis, candidates)

        with self.profile.timer("rerank_time"):
            scored: list[tuple[Chunk, ScoreComponents]] = []
            components_by_id: dict[str, ScoreComponents] = {}
            for chunk in candidates:
                recency = 1.0 if chunk.file_path in changed else 0.4 if chunk.file_path in recent else 0.0
                if chunk.file_path in related:
                    recency = max(recency, 0.35)
                text_for_terms = f"{chunk.file_path} {chunk.symbol_name or ''} {chunk.text}"
                if self.config.use_reranker:
                    components = ScoreComponents(
                        dense_similarity=dense_scores.get(chunk.chunk_id, 0.0),
                        lexical_score=lexical_scores.get(chunk.chunk_id, 0.0),
                        exact_term_overlap=_term_overlap(prompt_tokens, text_for_terms),
                        symbol_overlap=_symbol_overlap(analysis, chunk),
                        path_relevance=path_relevance(analysis, chunk.file_path),
                        filename_relevance=filename_relevance(analysis, chunk.file_path),
                        test_file_relevance=test_relevance(analysis, chunk),
                        import_dependency_relevance=0.0,
                        recency_relevance=recency,
                        config_relevance=config_relevance(analysis, chunk),
                        low_signal_penalty=low_signal_penalty(chunk),
                        generated_vendor_penalty=generated_vendor_penalty(chunk.file_path),
                    )
                else:
                    components = ScoreComponents(
                        dense_similarity=dense_scores.get(chunk.chunk_id, 0.0),
                        lexical_score=lexical_scores.get(chunk.chunk_id, 0.0),
                    )
                components_by_id[chunk.chunk_id] = components
                scored.append((chunk, components))
            self._graph_boost(analysis, candidates, components_by_id)
            ranked = rank_chunks(scored, self.weights)
        ranked = ranked[:top_k]
        self.profile.set("number_of_candidates_after_rerank", len(ranked))
        self.profile.set(
            "embeddings_mem_mb",
            sum((np.asarray(item.chunk.embedding).nbytes if item.chunk.embedding is not None else 0) for item in ranked)
            / (1024 * 1024),
        )
        if self.config.debug_ranking:
            for idx, item in enumerate(ranked[:20], start=1):
                print(f"Rank {idx}:")
                print(f"path: {item.chunk.file_path}:{item.chunk.start_line}-{item.chunk.end_line}")
                print(f"score: {item.score:.3f}")
                print(", ".join(f"{k}: {v:.3f}" for k, v in item.components.as_debug_dict().items()))
                print(f"reason: {item.reason}")
        return ranked
