from ctxmin.chunking import Chunk
from ctxmin.packing import pack_context
from ctxmin.prompt_analyzer import analyze_prompt
from ctxmin.ranking import RankedChunk, ScoreComponents


def _chunk(idx: int, tokens: int = 20) -> Chunk:
    text = "\n".join(f"line {i}" for i in range(tokens))
    return Chunk(
        chunk_id=f"c{idx}",
        repo_path="/repo",
        file_path=f"src/file{idx}.py",
        start_line=1,
        end_line=tokens,
        language="python",
        symbol_name=f"symbol_{idx}",
        chunk_type="function",
        text=text,
        token_estimate=tokens,
        content_hash=f"h{idx}",
    )


def test_packer_preserves_constraints_and_stays_under_budget():
    analysis = analyze_prompt("Fix src/file1.py. Must not call remote services.")
    ranked = [
        RankedChunk(_chunk(1, 30), 0.9, ScoreComponents(path_relevance=1), "explicit path"),
        RankedChunk(_chunk(2, 300), 0.8, ScoreComponents(dense_similarity=1), "semantic"),
    ]

    pack = pack_context(analysis, ranked, budget=180)

    assert pack.estimated_tokens <= 180
    assert any("remote services" in c for c in pack.analysis.hard_constraints)
    assert pack.chunks
    assert pack.chunks[0].ranked.chunk.file_path == "src/file1.py"
