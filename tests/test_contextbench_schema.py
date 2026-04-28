import json

from ctxmin.chunking import chunk_text
from ctxmin.cli import build_parser
from ctxmin.contextbench_eval import GOLD_ONLY_WARNING, GoldSpan, parse_gold_context, prepare_contextbench_repo
from ctxmin.metrics import span_metrics, span_overlap


def test_contextbench_gold_context_schema_from_json_string():
    raw = json.dumps(
        [
            {
                "file": "astropy/table/table.py",
                "start_line": 10,
                "end_line": 12,
                "content": "a\nb\nc",
            }
        ]
    )

    spans = parse_gold_context(raw)

    assert len(spans) == 1
    assert spans[0].file == "astropy/table/table.py"
    assert spans[0].start_line == 10
    assert spans[0].end_line == 12


def test_contextbench_cli_defaults_to_distractor_mode():
    parser = build_parser()

    args = parser.parse_args(["bench-contextbench"])

    assert args.mode == "distractor"


def test_gold_only_mode_is_explicit_smoke_warning(tmp_path):
    spans = [GoldSpan(file="pkg/parser.py", start_line=1, end_line=2, content="def parse():\n    return 1")]
    row = {"instance_id": "example-1", "language": "python"}

    prepared = prepare_contextbench_repo(row, spans, mode="gold-only", base_root=tmp_path)

    assert prepared.warning == GOLD_ONLY_WARNING
    assert (prepared.repo_path / "pkg" / "parser.py").exists()
    assert len(list(prepared.repo_path.rglob("*.py"))) == 1


def test_distractor_mode_adds_non_gold_chunks(tmp_path):
    spans = [
        GoldSpan(
            file="pkg/parser.py",
            start_line=1,
            end_line=3,
            content="def parse(value):\n    return value\n",
        )
    ]
    row = {
        "instance_id": "example-2",
        "language": "python",
        "problem_statement": "Fix parser behavior in pkg/parser.py",
    }

    prepared = prepare_contextbench_repo(row, spans, mode="distractor", base_root=tmp_path)

    all_chunks = []
    for path in prepared.repo_path.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".md", ".toml", ".yaml"}:
            rel = path.relative_to(prepared.repo_path).as_posix()
            all_chunks.extend(chunk_text(str(prepared.repo_path), rel, path.read_text(encoding="utf-8")))
    gold_span = ("pkg/parser.py", 1, 3)
    non_gold = [
        chunk
        for chunk in all_chunks
        if not span_overlap((chunk.file_path, chunk.start_line, chunk.end_line), gold_span)
    ]

    assert prepared.synthetic_distractor_files > 0
    assert len(non_gold) >= 10
    assert (prepared.repo_path / "docs" / "contextbench_distractor.md").exists()
    assert (prepared.repo_path / "pyproject.toml").exists()


def test_span_metrics_reports_precision_recall_f1():
    selected = {("a.py", 10, 20), ("b.py", 1, 5)}
    gold = {("a.py", 15, 18), ("c.py", 1, 2)}

    precision, recall, f1 = span_metrics(selected, gold)

    assert precision == 0.5
    assert recall == 0.5
    assert f1 == 0.5
