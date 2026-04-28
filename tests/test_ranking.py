from ctxmin.chunking import Chunk
from ctxmin.ranking import ScoreComponents, rank_chunks


def test_ranking_formula_prefers_path_match_over_plain_semantic_when_stronger():
    chunk_a = Chunk("a", "/repo", "src/a.py", 1, 2, "python", "target", "function", "def target(): pass", 4, "ha")
    chunk_b = Chunk("b", "/repo", "src/b.py", 1, 2, "python", "other", "function", "def other(): pass", 4, "hb")

    ranked = rank_chunks(
        [
            (chunk_a, ScoreComponents(dense_similarity=0.5, lexical_score=1.0, path_relevance=1.0)),
            (chunk_b, ScoreComponents(dense_similarity=1.0)),
        ]
    )

    assert ranked[0].chunk.chunk_id == "a"
    assert "path" in ranked[0].reason
