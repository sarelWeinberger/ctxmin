# ctxmin — Code Diagram

Visual map of the [ctxmin/](ctxmin/) package: how modules depend on each other, the main data types they exchange, and how a request flows through the system.

---

## 1. Module dependency graph

Arrows go from importer → imported. Leaves on the right (`config`, `utils`, `secret_scanner`) have no internal dependencies.

```mermaid
graph LR
  CLI[cli.py]
  CB[contextbench_eval.py]
  IDX[repo_indexer.py]
  RET[retrieval.py]
  PCK[packing.py]
  RND[render.py]
  RNK[ranking.py]
  CHK[chunking.py]
  PA[prompt_analyzer.py]
  STO[storage.py]
  MOD[models.py]
  PRF[profiling.py]
  MET[metrics.py]
  UTL[utils.py]
  CFG[config.py]
  SEC[secret_scanner.py]

  CLI --> CB
  CLI --> IDX
  CLI --> RET
  CLI --> PCK
  CLI --> RND
  CLI --> PA
  CLI --> MOD
  CLI --> PRF
  CLI --> STO
  CLI --> CFG

  CB --> IDX
  CB --> RET
  CB --> PCK
  CB --> PA
  CB --> MOD
  CB --> STO
  CB --> CHK
  CB --> MET
  CB --> PRF
  CB --> CFG
  CB --> UTL

  IDX --> CHK
  IDX --> STO
  IDX --> MOD
  IDX --> PRF
  IDX --> UTL
  IDX --> CFG

  RET --> CHK
  RET --> RNK
  RET --> STO
  RET --> MOD
  RET --> PA
  RET --> PRF
  RET --> UTL
  RET --> CFG

  PCK --> RNK
  PCK --> PA
  PCK --> UTL

  RND --> PCK

  RNK --> CHK
  RNK --> CFG

  CHK --> UTL
  CHK --> CFG

  PA --> SEC
  PA --> UTL

  STO --> CHK
  STO --> UTL
  STO --> CFG

  MOD --> CFG
  UTL --> CFG
```

---

## 2. Key types and relationships

Dataclasses and the few stateful classes, grouped by role.

```mermaid
classDiagram
  direction LR

  class Chunk {
    +chunk_id: str
    +file_path: str
    +start_line: int
    +end_line: int
    +language: str
    +symbol_name: str?
    +chunk_type: str
    +text: str
    +token_estimate: int
    +content_hash: str
    +embedding: ndarray?
    +cache_key: str?
  }

  class ScoreComponents {
    +dense_similarity: float
    +lexical_score: float
    +exact_term_overlap: float
    +symbol_overlap: float
    +path_relevance: float
    +filename_relevance: float
    +test_file_relevance: float
    +import_dependency_relevance: float
    +recency_relevance: float
    +config_relevance: float
    +redundancy_penalty: float
    +low_signal_penalty: float
    +generated_vendor_penalty: float
    +total(weights) float
    +reasons() list~str~
  }

  class RankedChunk {
    +chunk: Chunk
    +score: float
    +components: ScoreComponents
    +reason: str
  }

  class PackedChunk {
    +ranked: RankedChunk
    +included_tokens: int
  }

  class PackResult {
    +analysis: PromptAnalysis
    +budget: int
    +chunks: list~PackedChunk~
    +estimated_tokens: int
    +omitted_chunks: int
    +original_estimated_tokens: int
    +savings_ratio: float
  }

  class PromptAnalysis {
    +prompt: str
    +task_type: str
    +explicit_file_paths: list
    +symbols: list
    +failing_tests: list
    +error_messages: list
    +stack_traces: list
    +commands: list
    +hard_constraints: list
    +risky_destructive_intent: bool
    +secrets: list~SecretFinding~
    +query_text() str
  }

  class RankingWeights {
    <<frozen dataclass>>
  }
  class RetrievalConfig {
    <<frozen dataclass>>
  }
  class IndexConfig {
    <<frozen dataclass>>
  }

  class IndexReport {
    +files_seen / indexed / unchanged / skipped
    +chunks_indexed / reused / reembedded / invalidated
    +embedding_model / backend / dimension
  }

  class IndexStorage {
    -conn: sqlite3.Connection
    +index_path: Path
    +mark_file() / mark_skipped()
    +replace_chunks_for_file()
    +get_cached_embedding() / store_cached_embedding()
    +iter_chunk_embeddings()
    +load_chunks_by_ids()
    +search_fts(query, limit)
    +stats()
  }

  class RepoIndexer {
    +repo_path
    +config: IndexConfig
    +model: EmbeddingModel | HashEmbeddingModel
    +storage: IndexStorage
    +profile: Profile
    +build_or_update() IndexReport
  }

  class Retriever {
    +repo_path
    +model
    +weights: RankingWeights
    +config: RetrievalConfig
    +storage: IndexStorage
    +profile: Profile
    +retrieve(analysis, top_k) list~RankedChunk~
  }

  class EmbeddingModel {
    +model_name / batch_size / device / dtype
    +backend = "sentence-transformers"
    +dimension: int
    +load() classmethod
    +embed_query(text) ndarray
    +embed_documents(texts) ndarray
  }
  class HashEmbeddingModel {
    +backend = "hash"
    +embed_query(text)
    +embed_documents(texts)
  }

  class Profile {
    +enabled: bool
    +timer(key) ctx
    +set / incr
    +finish() dict
  }

  RepoIndexer --> IndexStorage
  RepoIndexer --> EmbeddingModel
  RepoIndexer --> IndexConfig
  RepoIndexer --> Profile
  RepoIndexer ..> Chunk : creates
  RepoIndexer ..> IndexReport : returns

  Retriever --> IndexStorage
  Retriever --> EmbeddingModel
  Retriever --> RankingWeights
  Retriever --> RetrievalConfig
  Retriever --> Profile
  Retriever ..> RankedChunk : returns

  RankedChunk *-- Chunk
  RankedChunk *-- ScoreComponents
  PackedChunk *-- RankedChunk
  PackResult o-- PackedChunk
  PackResult --> PromptAnalysis

  EmbeddingModel <|.. HashEmbeddingModel : duck-typed alt
```

---

## 3. Request flow — `ctxmin minimize` / `explain`

Top-level path from CLI invocation to rendered output. `_build_pack` is the shared core; `minimize` and `explain` only differ in how they render the resulting `PackResult`.

```mermaid
sequenceDiagram
  autonumber
  participant U as User
  participant CLI as cli.py
  participant PA as prompt_analyzer
  participant IX as RepoIndexer
  participant ST as IndexStorage (sqlite)
  participant EM as EmbeddingModel
  participant RT as Retriever
  participant RK as ranking.rank_chunks
  participant PK as packing.pack_context
  participant RD as render.*

  U->>CLI: ctxmin minimize PROMPT --repo PATH
  CLI->>CLI: _retrieval_config_from_args(args)
  CLI->>ST: IndexStorage(repo).stats()
  alt index empty
    CLI->>IX: RepoIndexer(repo).build_or_update()
    IX->>ST: scan/skip/cache embeddings
    IX->>EM: embed_documents(new chunks)
    IX-->>CLI: IndexReport
  end
  CLI->>PA: analyze_prompt(prompt)
  PA-->>CLI: PromptAnalysis
  CLI->>EM: EmbeddingModel.load()
  CLI->>RT: Retriever(repo, model, cfg).retrieve(analysis)
  RT->>EM: embed_query(analysis.query_text())
  RT->>ST: iter_chunk_embeddings()  (dense scan)
  RT->>ST: search_fts(query)        (BM25)
  RT->>ST: load_chunks_by_ids(merged)
  RT->>RK: rank_chunks(scored, weights)
  RK-->>RT: list~RankedChunk~
  RT-->>CLI: ranked
  CLI->>PK: pack_context(analysis, ranked, budget, MMR)
  PK-->>CLI: PackResult
  CLI->>RD: render_minimized_prompt(pack)  / render_explain(pack)
  RD-->>U: stdout
```

---

## 4. Indexing pipeline — `ctxmin index`

What happens inside `RepoIndexer.build_or_update()`.

```mermaid
flowchart TD
  A[Repo path] --> B{is git repo?}
  B -- yes --> C[git ls-files]
  B -- no --> D[walk_supported_files]
  C --> E[candidate file list]
  D --> E
  E --> F{skip_reason?<br/>skip_dir / unsupported_extension /<br/>lock_file / too_large / binary / unreadable / generated}
  F -- yes --> G[storage.mark_skipped]
  F -- no --> H[hash file bytes]
  H --> I{file_hash unchanged<br/>AND not force_reembed?}
  I -- yes --> J[reuse existing chunks<br/>chunks_reused++]
  I -- no --> K[chunk_text by language<br/>python AST / brace / markdown / config / line-window]
  K --> L{cache_key hit per chunk?}
  L -- yes --> M[load cached embedding<br/>chunks_reused++]
  L -- no --> N[model.embed_documents<br/>chunks_reembedded++]
  M --> O[storage.replace_chunks_for_file]
  N --> O
  O --> P[storage.mark_file + commit]
  P --> Q[remove_files_not_in seen]
  Q --> R[IndexReport]
```

Schema written by `IndexStorage._init_schema()`:

- `settings(key, value)` — embedding model/backend/dim, chunker_version, normalization_version
- `files(file_path, file_hash, language, indexed_at, skipped_reason)`
- `skipped_files(file_path, reason)`
- `chunks(chunk_id, file_path, start_line, end_line, language, symbol_name, chunk_type, text, token_estimate, content_hash, cache_key, embedding BLOB)`
- `embedding_cache(cache_key, model_name, model_revision, embedding_backend, embedding_dim, chunk_hash, chunker_version, normalization_version, file_path, file_hash, embedding BLOB, created_at)`
- `chunk_fts` — FTS5 virtual table over `(file_path, symbol_name, chunk_type, language, text)` with custom `tokenchars '_./-'`

---

## 5. Retrieval pipeline — inside `Retriever.retrieve()`

```mermaid
flowchart LR
  PA[PromptAnalysis] --> Q[query_text]
  Q --> DV[model.embed_query]
  DV --> DS[dense scan via<br/>iter_chunk_embeddings<br/>top_k by cosine]
  Q --> LX[_lexical_query<br/>OR-of-quoted-terms]
  LX --> FT[storage.search_fts<br/>BM25 + rank-position score]
  DS --> MG[_merge_candidates<br/>0.55 dense + 0.45 lexical<br/>cap = max_candidates]
  FT --> MG
  MG --> LD[storage.load_chunks_by_ids]
  LD --> SC[Per-chunk ScoreComponents:<br/>dense, lexical, exact_term_overlap,<br/>symbol_overlap, path_relevance,<br/>filename_relevance, test_file_relevance,<br/>recency, config, low_signal_penalty,<br/>generated_vendor_penalty]
  SC --> GB{config.graph_boost?}
  GB -- yes --> EX[expand to chunks referencing<br/>seed symbols, up to graph_expand_budget]
  GB -- no --> RK
  EX --> RK[rank_chunks: weighted sum<br/>via RankingWeights]
  RK --> TK[top_k = rerank_top_k]
  TK --> OUT[list of RankedChunk]
```

Then `pack_context` ([packing.py](ctxmin/packing.py)) takes the ranked list, optionally re-orders via MMR (`mmr_lambda`, embedding cosine vs token Jaccard fallback), and greedily fills the token budget — explicit-file-path matches get a small overflow allowance (+300 tokens).

---

## 6. ContextBench evaluation — `ctxmin bench-contextbench`

```mermaid
flowchart TD
  CMD["CLI: bench-contextbench<br/>--mode {gold-only, distractor, full-repo}"] --> EV[evaluate_contextbench]
  EV --> DS[load dataset<br/>default | contextbench_verified]
  DS --> PR[PreparedRepo<br/>(checkout / synth distractors)]
  PR --> IX2[RepoIndexer.build_or_update]
  IX2 --> RT2[Retriever.retrieve]
  RT2 --> PK2[pack_context]
  PK2 --> ME[metrics:<br/>file_metrics, span_metrics,<br/>span_overlap, token_savings]
  ME --> RPT[result JSON +<br/>optional Profile]
```

---

## File index (with line-anchored entry points)

- [ctxmin/cli.py](ctxmin/cli.py) — argparse entry point. `build_parser` [cli.py:155](ctxmin/cli.py#L155); `_build_pack` [cli.py:55](ctxmin/cli.py#L55); subcommand handlers [cli.py:20-152](ctxmin/cli.py#L20-L152)
- [ctxmin/config.py](ctxmin/config.py) — constants + frozen dataclasses `RankingWeights` [config.py:103](ctxmin/config.py#L103), `RetrievalConfig` [config.py:120](ctxmin/config.py#L120), `IndexConfig` [config.py:139](ctxmin/config.py#L139)
- [ctxmin/models.py](ctxmin/models.py) — `EmbeddingModel` [models.py:55](ctxmin/models.py#L55) and `HashEmbeddingModel` [models.py:20](ctxmin/models.py#L20) fallback
- [ctxmin/chunking.py](ctxmin/chunking.py) — `Chunk` [chunking.py:12](ctxmin/chunking.py#L12); language-specific chunkers; `chunk_text` dispatcher [chunking.py:219](ctxmin/chunking.py#L219)
- [ctxmin/storage.py](ctxmin/storage.py) — `IndexStorage` SQLite + FTS5 [storage.py:19](ctxmin/storage.py#L19)
- [ctxmin/repo_indexer.py](ctxmin/repo_indexer.py) — `RepoIndexer.build_or_update` [repo_indexer.py:110](ctxmin/repo_indexer.py#L110); `IndexReport` [repo_indexer.py:29](ctxmin/repo_indexer.py#L29)
- [ctxmin/retrieval.py](ctxmin/retrieval.py) — `Retriever.retrieve` [retrieval.py:272](ctxmin/retrieval.py#L272); per-feature scorers [retrieval.py:85-182](ctxmin/retrieval.py#L85-L182)
- [ctxmin/ranking.py](ctxmin/ranking.py) — `ScoreComponents` [ranking.py:9](ctxmin/ranking.py#L9), `rank_chunks` [ranking.py:97](ctxmin/ranking.py#L97)
- [ctxmin/packing.py](ctxmin/packing.py) — `pack_context` with MMR [packing.py:94](ctxmin/packing.py#L94)
- [ctxmin/render.py](ctxmin/render.py) — `render_minimized_prompt` [render.py:12](ctxmin/render.py#L12), `render_explain` [render.py:78](ctxmin/render.py#L78)
- [ctxmin/prompt_analyzer.py](ctxmin/prompt_analyzer.py) — regex-based `analyze_prompt`
- [ctxmin/secret_scanner.py](ctxmin/secret_scanner.py) — `scan_text` for credentials in prompts
- [ctxmin/profiling.py](ctxmin/profiling.py) — `Profile` timing + counters
- [ctxmin/metrics.py](ctxmin/metrics.py) — span/file overlap and token-savings metrics for benchmarks
- [ctxmin/contextbench_eval.py](ctxmin/contextbench_eval.py) — `evaluate_contextbench` orchestrator
- [ctxmin/utils.py](ctxmin/utils.py) — paths, hashing, git helpers, language detection
