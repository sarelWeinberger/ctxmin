from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .prompt_analyzer import PromptAnalysis
from .ranking import RankedChunk
from .utils import estimate_tokens, tokenize


@dataclass
class PackedChunk:
    ranked: RankedChunk
    included_tokens: int


@dataclass
class PackResult:
    analysis: PromptAnalysis
    budget: int
    chunks: list[PackedChunk] = field(default_factory=list)
    estimated_tokens: int = 0
    omitted_chunks: int = 0
    original_estimated_tokens: int = 0

    @property
    def savings_ratio(self) -> float:
        if self.original_estimated_tokens <= 0:
            return 0.0
        saved = max(0, self.original_estimated_tokens - self.estimated_tokens)
        return saved / self.original_estimated_tokens


def _critical_sections_tokens(analysis: PromptAnalysis) -> int:
    parts = [
        analysis.prompt,
        "\n".join(analysis.hard_constraints),
        "\n".join(analysis.error_messages),
        "\n".join(analysis.stack_traces),
        "\n".join(analysis.failing_tests),
        "\n".join(analysis.explicit_file_paths),
        "\n".join(analysis.symbols),
    ]
    return estimate_tokens("\n".join(parts))


def _chunk_overhead(ranked: RankedChunk) -> int:
    chunk = ranked.chunk
    return estimate_tokens(
        f"file: {chunk.file_path}\nlines: {chunk.start_line}-{chunk.end_line}\nsymbol: {chunk.symbol_name}\nreason: {ranked.reason}\n"
    ) + 8


def _similarity(a: RankedChunk, b: RankedChunk) -> float:
    ea = a.chunk.embedding
    eb = b.chunk.embedding
    if ea is not None and eb is not None:
        va = np.asarray(ea, dtype=np.float32)
        vb = np.asarray(eb, dtype=np.float32)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom:
            return max(0.0, min(1.0, float(va @ vb) / denom))
    toks_a = tokenize(f"{a.chunk.file_path} {a.chunk.symbol_name or ''} {a.chunk.text}")
    toks_b = tokenize(f"{b.chunk.file_path} {b.chunk.symbol_name or ''} {b.chunk.text}")
    if not toks_a or not toks_b:
        return 0.0
    return len(toks_a & toks_b) / len(toks_a | toks_b)


def _explicit_match(analysis: PromptAnalysis, ranked: RankedChunk) -> bool:
    chunk = ranked.chunk
    return any(path.endswith(chunk.file_path) or chunk.file_path.endswith(path) for path in analysis.explicit_file_paths)


def _mmr_order(candidates: list[RankedChunk], mmr_lambda: float, limit: int) -> list[RankedChunk]:
    pool = candidates[:limit]
    selected: list[RankedChunk] = []
    remaining = list(pool)
    while remaining:
        best_idx = 0
        best_score = -math.inf
        for idx, item in enumerate(remaining):
            redundancy = max((_similarity(item, chosen) for chosen in selected), default=0.0)
            adjusted = item.score - mmr_lambda * redundancy
            if adjusted > best_score:
                best_score = adjusted
                best_idx = idx
        selected.append(remaining.pop(best_idx))
    return selected + candidates[limit:]


def pack_context(
    analysis: PromptAnalysis,
    ranked_chunks: list[RankedChunk],
    budget: int = 6000,
    *,
    use_mmr: bool = True,
    mmr_lambda: float = 0.35,
    mmr_top_k: int = 80,
    min_score: float = 0.22,
) -> PackResult:
    critical_budget = _critical_sections_tokens(analysis) + 180
    result = PackResult(
        analysis=analysis,
        budget=budget,
        original_estimated_tokens=estimate_tokens(analysis.prompt)
        + sum(item.chunk.token_estimate for item in ranked_chunks),
    )
    used = min(critical_budget, budget)
    ordered = _mmr_order(ranked_chunks, mmr_lambda=mmr_lambda, limit=mmr_top_k) if use_mmr else ranked_chunks
    for ranked in ordered:
        chunk = ranked.chunk
        needed = chunk.token_estimate + _chunk_overhead(ranked)
        explicit = _explicit_match(analysis, ranked)
        if ranked.score < min_score and not explicit:
            result.omitted_chunks += 1
            continue
        if used + needed <= budget or (explicit and used + needed <= budget + 300):
            result.chunks.append(PackedChunk(ranked=ranked, included_tokens=needed))
            used += needed
        else:
            result.omitted_chunks += 1
    result.estimated_tokens = min(used, budget)
    return result
