# ctxmin

`ctxmin` is a local Context Minimization Gateway for coding agents such as Codex CLI and Claude Code. It indexes a repository, retrieves relevant code chunks for a developer prompt, and emits a compact structured prompt that preserves constraints, errors, failing tests, and the smallest relevant source context it can fit into a token budget.

The primary strategy is extractive selection, not generative summarization. Repo code and prompts stay on the developer machine.

## Install

```bash
pip install -e ".[embeddings,bench,dev]"
```

The default embedding model is `google/embeddinggemma-300m` loaded with `sentence-transformers`. `ctxmin` uses `encode_query` for prompts and `encode_document` for code/doc chunks when the installed model exposes those methods, and falls back to `encode` when necessary. CPU is the default first-class path; CUDA and MPS are used only when available.

For offline smoke tests without downloading the model:

```bash
CTXMIN_EMBEDDING_BACKEND=hash ctxmin index .
```

The hash backend is deterministic and local, but it is a fallback/test backend, not the intended production retriever.

## Commands

```bash
ctxmin index <repo_path>
ctxmin minimize "<developer prompt>" --repo <repo_path> --budget 6000
ctxmin explain "<developer prompt>" --repo <repo_path>
ctxmin bench-contextbench --dataset default --limit 20 --budget 6000
ctxmin bench-contextbench --dataset contextbench_verified --budget 6000
ctxmin bench-contextbench --mode gold-only --dataset default --limit 5
ctxmin bench-contextbench --mode full-repo --dataset default --limit 5 --budget 6000
ctxmin bench-contextbench --mode distractor --dataset default --limit 20 --budget 6000 --profile
ctxmin bench-contextbench --mode distractor --dataset default --limit 20 --budget 6000 --ablate-retrieval
ctxmin stats <repo_path>
```

## Storage

Indexes are stored locally under:

```text
~/.ctxmin/indexes/<repo_hash>/
```

Metadata and embeddings are stored in SQLite. FAISS can be added later as an optional acceleration layer; the default MVP vector search uses normalized numpy cosine similarity.

## What Gets Indexed

By default, `ctxmin` indexes git-tracked source-like files and skips heavy or generated paths such as `.git`, `node_modules`, `dist`, `build`, `target`, virtualenvs, caches, coverage output, large binaries, images, and lock files.

Supported file types include Python, JavaScript, TypeScript, Java, Go, Rust, C, C++, Markdown, YAML, JSON, and TOML.

## ContextBench

The benchmark command loads the Hugging Face ContextBench dataset configs (`default` and `contextbench_verified`) and supports three explicit modes:

- `gold-only`: smoke test only. This recreates the old behavior by building temporary repos from gold-context spans only. The CLI prints a warning and does not treat token savings as meaningful.
- `distractor`: default mode. This builds a temporary repo from gold context plus same-language, same-directory, similarly named, test, config, and docs distractors. It tries to create at least 10x more non-gold chunks than gold chunks when possible.
- `full-repo`: real full benchmark mode. When ContextBench rows include `repo_url` / `repo` and `base_commit`, this clones the original repo, checks out the commit, indexes the full repo, and compares selected context to gold context.

Only `full-repo` should be described as a real full repository benchmark.

The retrieval pipeline is memory-bounded by default:

- dense search streams vectors from SQLite and keeps only `--dense-top-k` candidates
- lexical search uses disk-backed SQLite FTS and keeps only `--lexical-top-k` candidates
- merged candidates are capped with `--max-candidates`
- reranking is capped with `--rerank-top-k`
- MMR packing runs only over `--mmr-top-k`
- optional graph boosting is disabled by default and bounded by `--graph-expand-budget`

Useful tuning flags:

```bash
ctxmin bench-contextbench \
  --mode distractor \
  --dataset default \
  --limit 20 \
  --budget 6000 \
  --dense-top-k 200 \
  --lexical-top-k 200 \
  --max-candidates 300 \
  --rerank-top-k 80 \
  --mmr-top-k 80 \
  --mmr-lambda 0.35 \
  --profile \
  --profile-json profile.json
```
