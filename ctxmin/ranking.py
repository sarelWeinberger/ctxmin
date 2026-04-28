from __future__ import annotations

from dataclasses import dataclass, field

from .chunking import Chunk
from .config import RankingWeights


@dataclass
class ScoreComponents:
    dense_similarity: float = 0.0
    lexical_score: float = 0.0
    exact_term_overlap: float = 0.0
    symbol_overlap: float = 0.0
    path_relevance: float = 0.0
    filename_relevance: float = 0.0
    test_file_relevance: float = 0.0
    import_dependency_relevance: float = 0.0
    recency_relevance: float = 0.0
    config_relevance: float = 0.0
    redundancy_penalty: float = 0.0
    low_signal_penalty: float = 0.0
    generated_vendor_penalty: float = 0.0

    def total(self, weights: RankingWeights | None = None) -> float:
        weights = weights or RankingWeights()
        return (
            weights.dense_similarity * self.dense_similarity
            + weights.lexical_score * self.lexical_score
            + weights.exact_term_overlap * self.exact_term_overlap
            + weights.symbol_overlap * self.symbol_overlap
            + weights.path_relevance * self.path_relevance
            + weights.filename_relevance * self.filename_relevance
            + weights.test_file_relevance * self.test_file_relevance
            + weights.import_dependency_relevance * self.import_dependency_relevance
            + weights.recency_relevance * self.recency_relevance
            + weights.config_relevance * self.config_relevance
            - weights.redundancy_penalty * self.redundancy_penalty
            - weights.low_signal_penalty * self.low_signal_penalty
            - weights.generated_vendor_penalty * self.generated_vendor_penalty
        )

    def reasons(self) -> list[str]:
        reasons: list[str] = []
        if self.path_relevance >= 0.75:
            reasons.append("strong path relevance")
        elif self.path_relevance > 0:
            reasons.append("path relevance")
        if self.filename_relevance >= 0.75:
            reasons.append("filename match")
        if self.symbol_overlap >= 0.5:
            reasons.append("symbol match")
        if self.exact_term_overlap >= 0.25:
            reasons.append("exact query term overlap")
        if self.lexical_score >= 0.25:
            reasons.append("lexical match")
        if self.dense_similarity >= 0.6:
            reasons.append("dense semantic match")
        if self.test_file_relevance > 0:
            reasons.append("test relevance")
        if self.import_dependency_relevance > 0:
            reasons.append("import or reference proximity")
        if self.config_relevance > 0:
            reasons.append("config/convention relevance")
        if self.generated_vendor_penalty > 0:
            reasons.append("penalized generated/vendor path")
        if self.low_signal_penalty > 0:
            reasons.append("penalized low-signal chunk")
        return reasons or ["ranked by combined retrieval score"]

    def as_debug_dict(self) -> dict[str, float]:
        return {
            "dense": self.dense_similarity,
            "lexical": self.lexical_score,
            "exact": self.exact_term_overlap,
            "symbol": self.symbol_overlap,
            "path": self.path_relevance,
            "filename": self.filename_relevance,
            "test": self.test_file_relevance,
            "import": self.import_dependency_relevance,
            "recency": self.recency_relevance,
            "config": self.config_relevance,
            "penalties": -(
                self.redundancy_penalty + self.low_signal_penalty + self.generated_vendor_penalty
            ),
        }


@dataclass
class RankedChunk:
    chunk: Chunk
    score: float
    components: ScoreComponents = field(default_factory=ScoreComponents)
    reason: str = ""


def rank_chunks(scored: list[tuple[Chunk, ScoreComponents]], weights: RankingWeights | None = None) -> list[RankedChunk]:
    ranked: list[RankedChunk] = []
    for chunk, components in scored:
        score = components.total(weights)
        ranked.append(
            RankedChunk(
                chunk=chunk,
                score=score,
                components=components,
                reason="; ".join(components.reasons()),
            )
        )
    ranked.sort(key=lambda item: (item.score, -item.chunk.token_estimate), reverse=True)
    return ranked
