from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import DEFAULT_TOKEN_BUDGET, RetrievalConfig
from .contextbench_eval import evaluate_contextbench
from .models import EmbeddingModel
from .packing import pack_context
from .profiling import Profile, render_profile, write_profile_json
from .prompt_analyzer import analyze_prompt
from .repo_indexer import RepoIndexer
from .render import render_explain, render_minimized_prompt
from .retrieval import Retriever
from .storage import IndexStorage


def _cmd_index(args: argparse.Namespace) -> int:
    indexer = RepoIndexer(args.repo_path)
    report = indexer.build_or_update()
    indexer.close()
    print(json.dumps(report.__dict__, indent=2, sort_keys=True))
    return 0


def _ensure_index(repo_path: str) -> None:
    storage = IndexStorage(repo_path)
    stats = storage.stats()
    storage.close()
    if int(stats["chunks"]) == 0:
        indexer = RepoIndexer(repo_path)
        indexer.build_or_update()
        indexer.close()


def _retrieval_config_from_args(args: argparse.Namespace) -> RetrievalConfig:
    return RetrievalConfig(
        dense_top_k=getattr(args, "dense_top_k", 200),
        lexical_top_k=getattr(args, "lexical_top_k", 200),
        max_candidates=getattr(args, "max_candidates", 300),
        rerank_top_k=getattr(args, "rerank_top_k", 80),
        mmr_top_k=getattr(args, "mmr_top_k", 80),
        mmr_lambda=getattr(args, "mmr_lambda", 0.35),
        use_mmr=not getattr(args, "no_mmr", False),
        graph_boost=bool(getattr(args, "graph_boost", False)) and not getattr(args, "disable_graph_boost", False),
        graph_expand_budget=getattr(args, "graph_expand_budget", 1000),
        debug_ranking=getattr(args, "debug_ranking", False),
        max_ram_mb=getattr(args, "max_ram_mb", None),
    )


def _build_pack(prompt: str, repo: str, budget: int, retrieval_config: RetrievalConfig | None = None, profile: Profile | None = None):
    _ensure_index(repo)
    analysis = analyze_prompt(prompt)
    model = EmbeddingModel.load()
    retriever = Retriever(repo, model=model, config=retrieval_config, profile=profile)
    ranked = retriever.retrieve(analysis, top_k=(retrieval_config.rerank_top_k if retrieval_config else 80))
    retriever.close()
    if profile:
        with profile.timer("mmr_pack_time"):
            return pack_context(
                analysis,
                ranked,
                budget=budget,
                use_mmr=retrieval_config.use_mmr if retrieval_config else True,
                mmr_lambda=retrieval_config.mmr_lambda if retrieval_config else 0.35,
                mmr_top_k=retrieval_config.mmr_top_k if retrieval_config else 80,
            )
    return pack_context(analysis, ranked, budget=budget)


def _cmd_minimize(args: argparse.Namespace) -> int:
    profile = Profile(enabled=args.profile or bool(args.profile_json))
    pack = _build_pack(args.prompt, args.repo, args.budget, _retrieval_config_from_args(args), profile)
    print(render_minimized_prompt(pack), end="")
    if profile.enabled:
        profile.set("number_of_selected_chunks", len(pack.chunks))
        profile.set("selected_chunk_count", len(pack.chunks))
        profile.set("selected_token_count", sum(p.ranked.chunk.token_estimate for p in pack.chunks))
        data = profile.finish()
        print(render_profile(data), file=sys.stderr, end="")
        if args.profile_json:
            write_profile_json(args.profile_json, data)
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    profile = Profile(enabled=args.profile or bool(args.profile_json))
    pack = _build_pack(args.prompt, args.repo, args.budget, _retrieval_config_from_args(args), profile)
    print(render_explain(pack), end="")
    if profile.enabled:
        profile.set("number_of_selected_chunks", len(pack.chunks))
        profile.set("selected_chunk_count", len(pack.chunks))
        profile.set("selected_token_count", sum(p.ranked.chunk.token_estimate for p in pack.chunks))
        data = profile.finish()
        print(render_profile(data), file=sys.stderr, end="")
        if args.profile_json:
            write_profile_json(args.profile_json, data)
    return 0


def _cmd_bench_contextbench(args: argparse.Namespace) -> int:
    profile = Profile(enabled=args.profile or bool(args.profile_json))
    result = evaluate_contextbench(
        dataset=args.dataset,
        limit=args.limit,
        budget=args.budget,
        mode=args.mode,
        retrieval_config=_retrieval_config_from_args(args),
        profile=profile,
        ablate_retrieval=args.ablate_retrieval,
    )
    if result.get("warning"):
        print(f"WARNING: {result['warning']}", file=sys.stderr)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.profile and result.get("profile"):
        print(render_profile(result["profile"]), file=sys.stderr, end="")
    if args.profile_json and result.get("profile"):
        write_profile_json(args.profile_json, result["profile"])
    return 0


def _add_retrieval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dense-top-k", type=int, default=200)
    parser.add_argument("--lexical-top-k", type=int, default=200)
    parser.add_argument("--max-candidates", type=int, default=300)
    parser.add_argument("--rerank-top-k", type=int, default=80)
    parser.add_argument("--mmr-top-k", type=int, default=80)
    parser.add_argument("--mmr-lambda", type=float, default=0.35)
    parser.add_argument("--no-mmr", action="store_true")
    parser.add_argument("--graph-boost", action="store_true")
    parser.add_argument("--disable-graph-boost", action="store_true")
    parser.add_argument("--graph-expand-budget", type=int, default=1000)
    parser.add_argument("--max-ram-mb", type=int, default=None)
    parser.add_argument("--debug-ranking", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-json")


def _cmd_stats(args: argparse.Namespace) -> int:
    storage = IndexStorage(args.repo_path)
    stats = storage.stats()
    skipped = storage.skipped_files(limit=25)
    storage.close()
    stats["skipped_file_examples"] = skipped
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ctxmin", description="Local extractive context minimization gateway.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Build or update a local repository index.")
    p_index.add_argument("repo_path")
    p_index.set_defaults(func=_cmd_index)

    p_min = sub.add_parser("minimize", help="Return a minimized prompt for a coding agent.")
    p_min.add_argument("prompt")
    p_min.add_argument("--repo", required=True)
    p_min.add_argument("--budget", type=int, default=DEFAULT_TOKEN_BUDGET)
    _add_retrieval_args(p_min)
    p_min.set_defaults(func=_cmd_minimize)

    p_exp = sub.add_parser("explain", help="Explain selected files/chunks and token savings.")
    p_exp.add_argument("prompt")
    p_exp.add_argument("--repo", required=True)
    p_exp.add_argument("--budget", type=int, default=DEFAULT_TOKEN_BUDGET)
    _add_retrieval_args(p_exp)
    p_exp.set_defaults(func=_cmd_explain)

    p_bench = sub.add_parser("bench-contextbench", help="Run a ContextBench evaluation.")
    p_bench.add_argument("--dataset", default="default", choices=["default", "contextbench_verified"])
    p_bench.add_argument(
        "--mode",
        default="distractor",
        choices=["gold-only", "distractor", "full-repo"],
        help="Evaluation mode. Default is distractor; gold-only is smoke-only.",
    )
    p_bench.add_argument("--limit", type=int, default=None)
    p_bench.add_argument("--budget", type=int, default=DEFAULT_TOKEN_BUDGET)
    p_bench.add_argument("--ablate-retrieval", action="store_true")
    _add_retrieval_args(p_bench)
    p_bench.set_defaults(func=_cmd_bench_contextbench)

    p_stats = sub.add_parser("stats", help="Show index statistics.")
    p_stats.add_argument("repo_path")
    p_stats.set_defaults(func=_cmd_stats)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"ctxmin: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
