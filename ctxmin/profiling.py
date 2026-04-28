from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


PROFILE_KEYS = [
    "scan_repo_time",
    "hash_files_time",
    "load_index_time",
    "embed_changed_chunks_time",
    "vector_search_time",
    "lexical_search_time",
    "candidate_merge_time",
    "rerank_time",
    "mmr_pack_time",
    "context_pack_time",
    "benchmark_eval_time",
    "total_time",
    "number_of_chunks_total",
    "number_of_chunks_embedded",
    "number_of_chunks_loaded_from_cache",
    "number_of_chunks_invalidated",
    "number_of_candidates_dense",
    "number_of_candidates_lexical",
    "number_of_candidates_after_merge",
    "number_of_candidates_after_rerank",
    "number_of_selected_chunks",
    "selected_token_count",
    "baseline_token_count",
    "peak_rss_mb",
    "embeddings_mem_mb",
    "lexical_index_mem_mb",
    "graph_index_mem_mb",
    "candidate_count",
    "selected_chunk_count",
]


@dataclass
class Profile:
    enabled: bool = False
    values: dict[str, float | int] = field(default_factory=dict)
    started_at: float = field(default_factory=time.perf_counter)

    def __post_init__(self) -> None:
        for key in PROFILE_KEYS:
            self.values.setdefault(key, 0)

    @contextmanager
    def timer(self, key: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add_time(key, time.perf_counter() - start)

    def add_time(self, key: str, seconds: float) -> None:
        self.values[key] = float(self.values.get(key, 0.0)) + seconds

    def incr(self, key: str, amount: int | float = 1) -> None:
        self.values[key] = self.values.get(key, 0) + amount

    def set(self, key: str, value: int | float) -> None:
        self.values[key] = value

    def merge(self, other: "Profile | dict[str, int | float] | None") -> None:
        if other is None:
            return
        source = other.values if isinstance(other, Profile) else other
        for key, value in source.items():
            if isinstance(value, (int, float)):
                self.values[key] = self.values.get(key, 0) + value

    def finish(self) -> dict[str, int | float]:
        self.values["peak_rss_mb"] = max(float(self.values.get("peak_rss_mb", 0)), current_rss_mb())
        self.values["total_time"] = time.perf_counter() - self.started_at
        return dict(self.values)


def render_profile(profile: dict[str, int | float]) -> str:
    lines = ["PROFILE"]
    for key in PROFILE_KEYS:
        value = profile.get(key, 0)
        if key.endswith("_time") or key == "total_time":
            lines.append(f"- {key}: {float(value):.3f}s")
        else:
            lines.append(f"- {key}: {int(value)}")
    return "\n".join(lines) + "\n"


def write_profile_json(path: str | Path, profile: dict[str, int | float]) -> None:
    Path(path).expanduser().write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def current_rss_mb() -> float:
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB, macOS reports bytes. This environment is Linux, but keep it sane.
        return rss / (1024.0 * 1024.0) if rss > 10_000_000 else rss / 1024.0
    except Exception:
        return 0.0
