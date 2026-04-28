from __future__ import annotations

import numpy as np

from ctxmin.config import RetrievalConfig
from ctxmin.contextbench_eval import evaluate_contextbench_ablations
from ctxmin.chunking import Chunk
from ctxmin.packing import pack_context
from ctxmin.profiling import PROFILE_KEYS, Profile
from ctxmin.prompt_analyzer import analyze_prompt
from ctxmin.ranking import RankedChunk, ScoreComponents, rank_chunks
from ctxmin.repo_indexer import RepoIndexer
from ctxmin.retrieval import Retriever, generated_vendor_penalty, path_relevance


class FakeEmbeddingModel:
    def __init__(self, name="fake-model", backend="hash", dimension=8):
        self.model_name = name
        self.backend = backend
        self.dimension = dimension
        self.model_revision = "test"

    def _vec(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        for idx, ch in enumerate(text.encode("utf-8")):
            vec[idx % self.dimension] += (ch % 17) / 17.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._vec(text) for text in texts]).astype(np.float32)


def test_embedding_cache_reuses_and_invalidates_backend_change(tmp_path):
    (tmp_path / "app.py").write_text("def target_symbol():\n    return 1\n", encoding="utf-8")

    first = RepoIndexer(tmp_path, model=FakeEmbeddingModel(name="m", backend="hash", dimension=8)).build_or_update()
    second = RepoIndexer(tmp_path, model=FakeEmbeddingModel(name="m", backend="hash", dimension=8)).build_or_update()
    third = RepoIndexer(tmp_path, model=FakeEmbeddingModel(name="m", backend="sentence-transformers", dimension=8)).build_or_update()

    assert first.chunks_reembedded > 0
    assert second.chunks_reembedded == 0
    assert second.chunks_reused > 0
    assert third.chunks_reembedded > 0
    assert third.chunks_invalidated > 0


def test_lexical_retrieval_returns_exact_symbol_match(tmp_path):
    (tmp_path / "app.py").write_text("def unique_symbol_abc(value):\n    return value\n", encoding="utf-8")
    RepoIndexer(tmp_path, model=FakeEmbeddingModel()).build_or_update()

    retriever = Retriever(
        tmp_path,
        model=FakeEmbeddingModel(),
        config=RetrievalConfig(use_dense=False, use_lexical=True, rerank_top_k=10),
    )
    ranked = retriever.retrieve(analyze_prompt("Fix unique_symbol_abc"), top_k=5)
    retriever.close()

    assert ranked
    assert ranked[0].chunk.symbol_name == "unique_symbol_abc"


def test_reranker_boosts_path_and_symbol_matches():
    analysis = analyze_prompt("Fix src/auth/session.py refresh_session")

    assert path_relevance(analysis, "src/auth/session.py") == 1.0
    assert generated_vendor_penalty("vendor/generated/client.py") > 0

    target = Chunk("a", "/repo", "src/auth/session.py", 1, 2, "python", "refresh_session", "function", "def refresh_session(): pass", 4, "ha")
    other = Chunk("b", "/repo", "dist/session_bundle.py", 1, 2, "python", "other", "function", "def other(): pass", 4, "hb")
    ranked = rank_chunks(
        [
            (target, ScoreComponents(dense_similarity=0.4, symbol_overlap=1.0, path_relevance=1.0, filename_relevance=1.0)),
            (other, ScoreComponents(dense_similarity=0.9, generated_vendor_penalty=1.0)),
        ]
    )

    assert ranked[0].chunk.chunk_id == "a"


def test_mmr_prefers_diverse_chunk_under_budget():
    analysis = analyze_prompt("Fix auth refresh")
    dup_a = Chunk("a", "/repo", "a.py", 1, 10, "python", "refresh", "function", "def refresh():\n    return 'a'", 20, "ha", embedding=np.array([1.0, 0.0], dtype=np.float32))
    dup_b = Chunk("b", "/repo", "b.py", 1, 10, "python", "refresh_copy", "function", "def refresh():\n    return 'b'", 20, "hb", embedding=np.array([1.0, 0.0], dtype=np.float32))
    diverse = Chunk("c", "/repo", "c.py", 1, 10, "python", "token", "function", "def token():\n    return 'c'", 20, "hc", embedding=np.array([0.0, 1.0], dtype=np.float32))
    ranked = [
        RankedChunk(dup_a, 0.9, ScoreComponents(dense_similarity=0.9), "a"),
        RankedChunk(dup_b, 0.85, ScoreComponents(dense_similarity=0.85), "b"),
        RankedChunk(diverse, 0.8, ScoreComponents(dense_similarity=0.8), "c"),
    ]

    pack = pack_context(analysis, ranked, budget=280, use_mmr=True, mmr_lambda=0.5, mmr_top_k=3)

    selected = [item.ranked.chunk.chunk_id for item in pack.chunks[:2]]
    assert selected == ["a", "c"]


def test_profile_contains_expected_timing_and_memory_keys():
    profile = Profile(enabled=True)
    data = profile.finish()

    for key in [
        "scan_repo_time",
        "vector_search_time",
        "lexical_search_time",
        "rerank_time",
        "mmr_pack_time",
        "peak_rss_mb",
        "embeddings_mem_mb",
        "lexical_index_mem_mb",
        "graph_index_mem_mb",
    ]:
        assert key in PROFILE_KEYS
        assert key in data


def test_ablation_rows_can_be_rendered(monkeypatch):
    import ctxmin.contextbench_eval as cb

    def fake_evaluate_contextbench(**kwargs):
        return {
            "averages": {
                "file_precision": 1.0,
                "file_recall": 1.0,
                "file_f1": 1.0,
                "span_precision": 1.0,
                "span_recall": 1.0,
                "span_f1": 1.0,
                "selected_tokens": 10,
                "baseline_tokens": 100,
                "token_savings_ratio": 0.9,
            },
            "profile": {
                "total_time": 1,
                "embed_changed_chunks_time": 0,
                "lexical_search_time": 0,
                "rerank_time": 0,
                "mmr_pack_time": 0,
                "peak_rss_mb": 1,
                "number_of_selected_chunks": 1,
                "candidate_count": 1,
            },
        }

    monkeypatch.setattr(cb, "evaluate_contextbench", fake_evaluate_contextbench)
    result = evaluate_contextbench_ablations(limit=1)

    assert len(result["rows"]) == 6
    assert {row["name"] for row in result["rows"]} >= {"dense only", "lexical only"}
