from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    file_precision: float
    file_recall: float
    file_f1: float
    span_precision: float
    span_recall: float
    span_f1: float
    selected_tokens: int
    baseline_tokens: int
    token_savings_ratio: float | None
    distractor_chunks: int


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def file_metrics(selected_files: set[str], gold_files: set[str]) -> tuple[float, float, float]:
    if not selected_files and not gold_files:
        return 1.0, 1.0, 1.0
    precision = len(selected_files & gold_files) / len(selected_files) if selected_files else 0.0
    recall = len(selected_files & gold_files) / len(gold_files) if gold_files else 0.0
    return precision, recall, _f1(precision, recall)


def span_overlap(selected: tuple[str, int, int], gold: tuple[str, int, int]) -> bool:
    sf, ss, se = selected
    gf, gs, ge = gold
    return sf == gf and max(ss, gs) <= min(se, ge)


def span_recall(selected_spans: set[tuple[str, int, int]], gold_spans: set[tuple[str, int, int]]) -> float:
    if not gold_spans:
        return 1.0
    hits = 0
    for gold in gold_spans:
        if any(span_overlap(selected, gold) for selected in selected_spans):
            hits += 1
    return hits / len(gold_spans)


def span_metrics(
    selected_spans: set[tuple[str, int, int]],
    gold_spans: set[tuple[str, int, int]],
) -> tuple[float, float, float]:
    if not selected_spans and not gold_spans:
        return 1.0, 1.0, 1.0
    precision_hits = 0
    for selected in selected_spans:
        if any(span_overlap(selected, gold) for gold in gold_spans):
            precision_hits += 1
    recall_hits = 0
    for gold in gold_spans:
        if any(span_overlap(selected, gold) for selected in selected_spans):
            recall_hits += 1
    precision = precision_hits / len(selected_spans) if selected_spans else 0.0
    recall = recall_hits / len(gold_spans) if gold_spans else 0.0
    return precision, recall, _f1(precision, recall)


def token_savings(selected_tokens: int, baseline_tokens: int) -> float:
    if baseline_tokens <= 0:
        return 0.0
    return max(0.0, 1.0 - (selected_tokens / baseline_tokens))
