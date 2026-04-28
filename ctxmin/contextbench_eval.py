from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .chunking import Chunk
from .config import CONTEXTBENCH_ROOT, DEFAULT_TOKEN_BUDGET, IndexConfig, RetrievalConfig
from .metrics import RetrievalMetrics, file_metrics, span_metrics, span_overlap, token_savings
from .models import EmbeddingModel
from .packing import pack_context
from .profiling import Profile
from .prompt_analyzer import analyze_prompt
from .repo_indexer import RepoIndexer
from .retrieval import Retriever
from .storage import IndexStorage
from .utils import language_for_path, sha256_text

GOLD_ONLY_WARNING = (
    "This is not a representative benchmark. Token savings and retrieval metrics are "
    "inflated/not representative."
)
EVAL_MODES = {"gold-only", "distractor", "full-repo"}


@dataclass
class GoldSpan:
    file: str
    start_line: int
    end_line: int
    content: str


@dataclass
class PreparedRepo:
    repo_path: Path
    mode: str
    warning: str | None = None
    synthetic_distractor_files: int = 0
    checkout_commit: str | None = None
    repo_url: str | None = None


def parse_gold_context(value: Any) -> list[GoldSpan]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        value = json.loads(value)
    spans = []
    for item in value:
        spans.append(
            GoldSpan(
                file=str(item["file"]),
                start_line=int(item["start_line"]),
                end_line=int(item["end_line"]),
                content=str(item.get("content", "")),
            )
        )
    return spans


def _load_contextbench(dataset: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install ctxmin with the bench extra: pip install -e '.[bench]'") from exc
    errors: list[Exception] = []
    for name in ("Contextbench/ContextBench", "Schwerli/ContextBench"):
        try:
            return load_dataset(name, dataset, split="train")
        except Exception as exc:
            errors.append(exc)
    raise RuntimeError(f"Could not load ContextBench dataset config {dataset!r}: {errors[-1]}")


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return safe[:160] or sha256_text(value)[:16]


def _row_id(row: dict[str, Any], fallback_index: int = 0) -> str:
    return str(row.get("instance_id") or row.get("original_inst_id") or f"row-{fallback_index}")


def _row_language(row: dict[str, Any], spans: list[GoldSpan]) -> str:
    if row.get("language"):
        return str(row["language"]).lower()
    for span in spans:
        language = language_for_path(span.file)
        if language:
            return language
    return "python"


def _language_extension(language: str) -> str:
    return {
        "python": ".py",
        "javascript": ".js",
        "typescript": ".ts",
        "java": ".java",
        "go": ".go",
        "rust": ".rs",
        "c": ".c",
        "cpp": ".cpp",
        "markdown": ".md",
        "yaml": ".yaml",
        "json": ".json",
        "toml": ".toml",
    }.get(language.lower(), ".py")


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def _write_gold_files(repo: Path, spans: list[GoldSpan]) -> None:
    by_file: dict[str, list[GoldSpan]] = {}
    for span in spans:
        by_file.setdefault(span.file, []).append(span)
    for file_path, file_spans in by_file.items():
        path = repo / file_path
        max_line = max(span.end_line for span in file_spans)
        lines = [""] * max_line
        for span in file_spans:
            content_lines = span.content.splitlines()
            for idx, content_line in enumerate(content_lines, start=span.start_line):
                if 1 <= idx <= len(lines):
                    lines[idx - 1] = content_line
        _write_file(path, "\n".join(lines))


def _name_token(path: str, index: int) -> str:
    stem = re.sub(r"[^A-Za-z0-9_]+", "_", Path(path).stem).strip("_") or "context"
    if stem[0].isdigit():
        stem = f"ctx_{stem}"
    return f"{stem}_distractor_{index}"


def _distractor_text(language: str, name: str, index: int, terms: list[str], chunks: int = 6) -> str:
    joined_terms = " ".join(terms[:12])
    language = language.lower()
    if language == "python":
        sections = [f'"""Synthetic ContextBench distractor for {joined_terms}."""']
        sections.append(f"class {name.title().replace('_', '')}Helper:")
        sections.append("    def __init__(self):\n        self.enabled = True")
        for n in range(chunks):
            sections.append(
                f"def {name}_candidate_{n}(value):\n"
                f"    marker = '{joined_terms} candidate {index} {n}'\n"
                "    if value is None:\n"
                "        return marker\n"
                "    return f'{marker}:{value}'"
            )
        return "\n\n".join(sections)
    if language in {"javascript", "typescript"}:
        export = "export " if language == "typescript" else ""
        return "\n\n".join(
            [
                f"{export}function {name}Candidate{n}(value) {{\n"
                f"  const marker = '{joined_terms} candidate {index} {n}';\n"
                "  return value == null ? marker : `${marker}:${value}`;\n"
                "}"
                for n in range(chunks)
            ]
        )
    if language == "java":
        methods = "\n".join(
            f"  public String {name}Candidate{n}(String value) {{ return \"{joined_terms}-{n}:\" + value; }}"
            for n in range(chunks)
        )
        return f"public class {name.title().replace('_', '')}Distractor {{\n{methods}\n}}"
    if language == "go":
        return "\n\n".join(
            [
                f"func {name.title().replace('_', '')}Candidate{n}(value string) string {{\n"
                f"    return \"{joined_terms}-{n}:\" + value\n"
                "}"
                for n in range(chunks)
            ]
        )
    if language == "rust":
        return "\n\n".join(
            [
                f"pub fn {name}_candidate_{n}(value: &str) -> String {{\n"
                f"    format!(\"{joined_terms}-{n}:{{}}\", value)\n"
                "}"
                for n in range(chunks)
            ]
        )
    if language in {"c", "cpp"}:
        return "\n\n".join(
            [
                f"int {name}_candidate_{n}(int value) {{\n"
                f"    int marker = {index + n};\n"
                "    return value + marker;\n"
                "}"
                for n in range(chunks)
            ]
        )
    if language == "markdown":
        return "\n\n".join(f"## {name} candidate {n}\n\nDistractor notes for {joined_terms}." for n in range(chunks))
    if language == "json":
        entries = ",\n".join(f'  "{name}_candidate_{n}": "{joined_terms} {n}"' for n in range(chunks))
        return "{\n" + entries + "\n}"
    if language == "toml":
        return "\n".join(f'{name}_candidate_{n} = "{joined_terms} {n}"' for n in range(chunks))
    return "\n".join(f"{name}_candidate_{n}: {joined_terms} {n}" for n in range(chunks))


def _distractor_terms(row: dict[str, Any], spans: list[GoldSpan]) -> list[str]:
    terms: list[str] = []
    for span in spans[:8]:
        terms.extend(part for part in re.split(r"[^A-Za-z0-9_]+", Path(span.file).stem) if part)
    prompt = str(row.get("problem_statement") or "")
    terms.extend(re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", prompt)[:20])
    return terms or ["context", "bench", "distractor"]


def _write_distractors(
    repo: Path,
    row: dict[str, Any],
    spans: list[GoldSpan],
    language: str,
    target_non_gold_chunks: int,
    offset: int = 0,
) -> int:
    ext = _language_extension(language)
    terms = _distractor_terms(row, spans)
    paths: list[Path] = []
    for idx, span in enumerate(spans[: max(1, min(len(spans), 12))], start=offset):
        gold_path = Path(span.file)
        parent = gold_path.parent
        stem = gold_path.stem
        paths.append(repo / parent / f"{stem}_same_dir_distractor_{idx}{ext}")
        paths.append(repo / parent / f"{stem}_candidate{ext}")
        paths.append(repo / parent / f"{stem}_v2{ext}")

    files_needed = max(8, math.ceil(target_non_gold_chunks / 4))
    for idx in range(offset, offset + files_needed):
        paths.append(repo / "distractors" / language / f"same_language_{idx}{ext}")

    for idx, stem in enumerate(["context", "selectors", "ranking", "packing"], start=offset):
        paths.append(repo / "tests" / f"test_{stem}_distractor{ext if ext != '.java' else '.java'}")

    paths.extend(
        [
            repo / "docs" / "contextbench_distractor.md",
            repo / "README_distractor.md",
            repo / "pyproject.toml",
            repo / ".github" / "workflows" / "contextbench_distractor.yaml",
        ]
    )

    written = 0
    seen: set[Path] = set()
    for idx, path in enumerate(paths, start=offset):
        if path in seen or path.exists():
            continue
        seen.add(path)
        path_language = language_for_path(path.name) or language
        name = _name_token(path.as_posix(), idx)
        content = _distractor_text(path_language, name, idx, terms, chunks=6)
        _write_file(path, content)
        written += 1
    return written


def _write_synthetic_repo(row: dict[str, Any], spans: list[GoldSpan], mode: str, base_root: Path) -> PreparedRepo:
    instance_id = _safe_id(_row_id(row))
    repo = base_root / f"{mode}_repos" / instance_id
    if repo.exists():
        shutil.rmtree(repo)
    repo.mkdir(parents=True, exist_ok=True)
    _write_gold_files(repo, spans)
    if mode == "gold-only":
        return PreparedRepo(repo_path=repo, mode=mode, warning=GOLD_ONLY_WARNING)
    language = _row_language(row, spans)
    target = max(20, 10 * max(1, len(spans)))
    files = _write_distractors(repo, row, spans, language, target_non_gold_chunks=target)
    return PreparedRepo(repo_path=repo, mode=mode, synthetic_distractor_files=files)


def _repo_url(row: dict[str, Any]) -> str | None:
    if row.get("repo_url"):
        return str(row["repo_url"])
    repo = row.get("repo")
    if repo and "/" in str(repo):
        return f"https://github.com/{repo}.git"
    return None


def _run_git(args: list[str], cwd: Path | None = None) -> None:
    proc = subprocess.run(["git", *args], cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"git {' '.join(args)} failed")


def _prepare_full_repo(row: dict[str, Any], base_root: Path) -> PreparedRepo:
    repo_url = _repo_url(row)
    if not repo_url:
        raise RuntimeError("ContextBench row does not include repo_url or repo metadata")
    commit = row.get("base_commit")
    repo_key = _safe_id(f"{repo_url}-{commit or 'default'}")
    repo = base_root / "full_repos" / repo_key
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        _run_git(["clone", repo_url, str(repo)])
    if commit:
        try:
            _run_git(["fetch", "--depth", "1", "origin", str(commit)], cwd=repo)
        except RuntimeError:
            _run_git(["fetch", "--all", "--tags"], cwd=repo)
        _run_git(["checkout", "--force", str(commit)], cwd=repo)
    return PreparedRepo(repo_path=repo, mode="full-repo", checkout_commit=str(commit) if commit else None, repo_url=repo_url)


def prepare_contextbench_repo(
    row: dict[str, Any],
    spans: list[GoldSpan],
    mode: str = "distractor",
    base_root: Path = CONTEXTBENCH_ROOT,
) -> PreparedRepo:
    if mode not in EVAL_MODES:
        raise ValueError(f"Unknown ContextBench mode {mode!r}; expected one of {sorted(EVAL_MODES)}")
    if mode == "full-repo":
        return _prepare_full_repo(row, base_root)
    return _write_synthetic_repo(row, spans, mode, base_root)


def _gold_span_set(spans: list[GoldSpan]) -> set[tuple[str, int, int]]:
    return {(span.file, span.start_line, span.end_line) for span in spans}


def _gold_file_set(spans: list[GoldSpan]) -> set[str]:
    return {span.file for span in spans}


def _selected_spans(pack) -> set[tuple[str, int, int]]:
    return {
        (packed.ranked.chunk.file_path, packed.ranked.chunk.start_line, packed.ranked.chunk.end_line)
        for packed in pack.chunks
    }


def _selected_tokens(pack) -> int:
    return sum(packed.ranked.chunk.token_estimate for packed in pack.chunks)


def _baseline_tokens(chunks: list[Chunk]) -> int:
    return sum(chunk.token_estimate for chunk in chunks)


def _count_gold_and_distractor_chunks(
    chunks: list[Chunk],
    gold_spans: set[tuple[str, int, int]],
) -> tuple[int, int]:
    gold_chunks = 0
    distractor_chunks = 0
    for chunk in chunks:
        selected = (chunk.file_path, chunk.start_line, chunk.end_line)
        if any(span_overlap(selected, gold) for gold in gold_spans):
            gold_chunks += 1
        else:
            distractor_chunks += 1
    return gold_chunks, distractor_chunks


def _index_repo(repo: Path, model, profile: Profile | None = None) -> list[Chunk]:
    indexer = RepoIndexer(repo, config=IndexConfig(embed_batch_size=16), model=model, profile=profile)
    indexer.build_or_update()
    indexer.close()
    storage = IndexStorage(repo)
    chunks = storage.load_chunks()
    storage.close()
    return chunks


def _run_instance(
    row: dict[str, Any],
    spans: list[GoldSpan],
    mode: str,
    budget: int,
    model,
    base_root: Path,
    fallback_index: int,
    retrieval_config: RetrievalConfig | None = None,
    profile: Profile | None = None,
) -> dict[str, Any]:
    profile = profile or Profile(enabled=False)
    retrieval_config = retrieval_config or RetrievalConfig()
    prepared = prepare_contextbench_repo(row, spans, mode=mode, base_root=base_root)
    chunks = _index_repo(prepared.repo_path, model, profile=profile)
    gold_spans = _gold_span_set(spans)
    if mode == "distractor":
        gold_chunk_count, distractor_count = _count_gold_and_distractor_chunks(chunks, gold_spans)
        target = 10 * max(1, gold_chunk_count)
        if distractor_count < target:
            language = _row_language(row, spans)
            _write_distractors(
                prepared.repo_path,
                row,
                spans,
                language,
                target_non_gold_chunks=target - distractor_count,
                offset=10_000,
            )
            chunks = _index_repo(prepared.repo_path, model, profile=profile)

    prompt = row.get("problem_statement") or row.get("issue") or row.get("prompt") or ""
    analysis = analyze_prompt(str(prompt))
    retriever = Retriever(prepared.repo_path, model=model, config=retrieval_config, profile=profile)
    ranked = retriever.retrieve(analysis, top_k=retrieval_config.rerank_top_k)
    retriever.close()
    with profile.timer("mmr_pack_time"):
        pack = pack_context(
            analysis,
            ranked,
            budget=budget,
            use_mmr=retrieval_config.use_mmr,
            mmr_lambda=retrieval_config.mmr_lambda,
            mmr_top_k=retrieval_config.mmr_top_k,
        )
    profile.incr("number_of_selected_chunks", len(pack.chunks))
    profile.incr("selected_chunk_count", len(pack.chunks))

    selected_files = {packed.ranked.chunk.file_path for packed in pack.chunks}
    selected_spans = _selected_spans(pack)
    gold_files = _gold_file_set(spans)
    file_precision, file_recall, file_f1 = file_metrics(selected_files, gold_files)
    span_precision, span_recall_value, span_f1 = span_metrics(selected_spans, gold_spans)
    selected_token_count = _selected_tokens(pack)
    baseline_token_count = _baseline_tokens(chunks)
    profile.incr("selected_token_count", selected_token_count)
    profile.incr("baseline_token_count", baseline_token_count)
    gold_chunk_count, distractor_count = _count_gold_and_distractor_chunks(chunks, gold_spans)
    if mode == "gold-only":
        distractor_count = 0
    savings = None if mode == "gold-only" else token_savings(selected_token_count, baseline_token_count)

    metrics = RetrievalMetrics(
        file_precision=file_precision,
        file_recall=file_recall,
        file_f1=file_f1,
        span_precision=span_precision,
        span_recall=span_recall_value,
        span_f1=span_f1,
        selected_tokens=selected_token_count,
        baseline_tokens=baseline_token_count,
        token_savings_ratio=savings,
        distractor_chunks=distractor_count,
    )
    return {
        "instance_id": _row_id(row, fallback_index=fallback_index),
        "repo": row.get("repo"),
        "repo_url": prepared.repo_url,
        "checkout_commit": prepared.checkout_commit,
        "mode": mode,
        "warning": prepared.warning,
        "token_savings_meaningful": mode != "gold-only",
        **metrics.__dict__,
        "selected_files": sorted(selected_files),
        "gold_files": sorted(gold_files),
        "selected_spans": sorted(selected_spans),
        "gold_spans": sorted(gold_spans),
        "gold_chunks": gold_chunk_count,
        "synthetic_distractor_files": prepared.synthetic_distractor_files,
    }


def _average_instances(instances: list[dict[str, Any]]) -> dict[str, Any]:
    metric_keys = [
        "file_precision",
        "file_recall",
        "file_f1",
        "span_precision",
        "span_recall",
        "span_f1",
        "selected_tokens",
        "baseline_tokens",
        "gold_chunks",
        "distractor_chunks",
    ]
    averages: dict[str, Any] = {}
    n = max(1, len(instances))
    for key in metric_keys:
        averages[key] = sum(float(instance[key]) for instance in instances) / n
    savings_values = [
        float(instance["token_savings_ratio"])
        for instance in instances
        if instance.get("token_savings_ratio") is not None
    ]
    averages["token_savings_ratio"] = (
        sum(savings_values) / len(savings_values) if savings_values else None
    )
    return averages


def evaluate_contextbench(
    dataset: str = "default",
    limit: int | None = 20,
    budget: int = DEFAULT_TOKEN_BUDGET,
    mode: str = "distractor",
    retrieval_config: RetrievalConfig | None = None,
    profile: Profile | None = None,
    ablate_retrieval: bool = False,
) -> dict[str, Any]:
    if mode not in EVAL_MODES:
        raise ValueError(f"Unknown ContextBench mode {mode!r}; expected one of {sorted(EVAL_MODES)}")
    if ablate_retrieval:
        return evaluate_contextbench_ablations(dataset=dataset, limit=limit, budget=budget, mode=mode)
    profile = profile or Profile(enabled=False)
    retrieval_config = retrieval_config or RetrievalConfig()
    with profile.timer("benchmark_eval_time"):
        ds = _load_contextbench(dataset)
    rows = list(ds.select(range(min(limit, len(ds))))) if limit else list(ds)
    per_instance: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    model = EmbeddingModel.load(allow_fallback=True)
    for index, row in enumerate(rows):
        row_dict = dict(row)
        spans = parse_gold_context(row_dict.get("gold_context"))
        if not spans:
            skipped.append({"instance_id": _row_id(row_dict, index), "reason": "missing_gold_context"})
            continue
        try:
            per_instance.append(
                _run_instance(
                    row_dict,
                    spans,
                    mode=mode,
                    budget=budget,
                    model=model,
                    base_root=CONTEXTBENCH_ROOT,
                    fallback_index=index,
                    retrieval_config=retrieval_config,
                    profile=profile,
                )
            )
        except Exception as exc:
            if mode == "full-repo":
                skipped.append({"instance_id": _row_id(row_dict, index), "reason": str(exc)})
                continue
            raise

    profile.set("number_of_chunks_total", max(profile.values.get("number_of_chunks_total", 0), 0))
    profile_values = profile.finish() if profile.enabled else None
    result: dict[str, Any] = {
        "dataset": dataset,
        "mode": mode,
        "benchmark_kind": "real full benchmark" if mode == "full-repo" else "synthetic benchmark",
        "limit": limit,
        "budget": budget,
        "instances_evaluated": len(per_instance),
        "instances_skipped": len(skipped),
        "skipped": skipped,
        "token_savings_meaningful": mode != "gold-only",
        "averages": _average_instances(per_instance),
        "instances": per_instance,
    }
    if profile_values is not None:
        result["profile"] = profile_values
    if mode == "gold-only":
        result["warning"] = GOLD_ONLY_WARNING
        result["benchmark_kind"] = "smoke test only"
    elif mode == "distractor":
        result["benchmark_kind"] = "synthetic distractor benchmark"
    return result


def evaluate_contextbench_ablations(
    dataset: str = "default",
    limit: int | None = 20,
    budget: int = DEFAULT_TOKEN_BUDGET,
    mode: str = "distractor",
) -> dict[str, Any]:
    variants = [
        ("dense only", RetrievalConfig(use_dense=True, use_lexical=False, use_reranker=True, use_mmr=False)),
        ("lexical only", RetrievalConfig(use_dense=False, use_lexical=True, use_reranker=True, use_mmr=False)),
        ("dense + lexical", RetrievalConfig(use_dense=True, use_lexical=True, use_reranker=False, use_mmr=False)),
        ("dense + lexical + reranker", RetrievalConfig(use_dense=True, use_lexical=True, use_reranker=True, use_mmr=False)),
        ("dense + lexical + reranker + MMR", RetrievalConfig(use_dense=True, use_lexical=True, use_reranker=True, use_mmr=True)),
        (
            "dense + lexical + reranker + MMR + graph boost",
            RetrievalConfig(use_dense=True, use_lexical=True, use_reranker=True, use_mmr=True, graph_boost=True),
        ),
    ]
    rows = []
    for name, config in variants:
        profile = Profile(enabled=True)
        result = evaluate_contextbench(
            dataset=dataset,
            limit=limit,
            budget=budget,
            mode=mode,
            retrieval_config=config,
            profile=profile,
            ablate_retrieval=False,
        )
        avg = result["averages"]
        rows.append(
            {
                "name": name,
                "file_precision": avg["file_precision"],
                "file_recall": avg["file_recall"],
                "file_f1": avg["file_f1"],
                "span_precision": avg["span_precision"],
                "span_recall": avg["span_recall"],
                "span_f1": avg["span_f1"],
                "avg_selected_tokens": avg["selected_tokens"],
                "avg_baseline_tokens": avg["baseline_tokens"],
                "token_savings_ratio": avg["token_savings_ratio"],
                "total_runtime": result.get("profile", {}).get("total_time", 0),
                "selection_runtime": sum(
                    result.get("profile", {}).get(key, 0)
                    for key in ["vector_search_time", "lexical_search_time", "candidate_merge_time", "rerank_time", "mmr_pack_time", "context_pack_time"]
                ),
                "embedding_time": result.get("profile", {}).get("embed_changed_chunks_time", 0),
                "lexical_search_time": result.get("profile", {}).get("lexical_search_time", 0),
                "rerank_time": result.get("profile", {}).get("rerank_time", 0),
                "mmr_time": result.get("profile", {}).get("mmr_pack_time", 0),
                "peak_rss_mb": result.get("profile", {}).get("peak_rss_mb", 0),
                "number_of_selected_chunks": result.get("profile", {}).get("number_of_selected_chunks", 0),
                "number_of_candidates": result.get("profile", {}).get("candidate_count", 0),
            }
        )
    return {
        "dataset": dataset,
        "mode": mode,
        "limit": limit,
        "budget": budget,
        "ablation": "retrieval",
        "rows": rows,
    }
