from __future__ import annotations

import re
from dataclasses import dataclass, field

from .secret_scanner import SecretFinding, scan_text
from .utils import dedupe_preserve_order

TASK_TYPES = {
    "bug_fix",
    "feature",
    "refactor",
    "test_failure",
    "explanation",
    "infra",
    "security",
    "unknown",
}

PATH_RE = re.compile(
    r"(?<![\w.-])(?:\.{1,2}/)?(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.@-]+\.[A-Za-z0-9]+(?::\d+(?:-\d+)?)?"
    r"|(?<![\w.-])[A-Za-z0-9_.@-]+\.(?:py|js|jsx|ts|tsx|java|go|rs|c|cc|cpp|cxx|h|hpp|md|yml|yaml|json|toml)(?::\d+(?:-\d+)?)?"
)
PY_STACK_RE = re.compile(r"Traceback \(most recent call last\):[\s\S]+?(?=\n\S|\Z)")
JS_STACK_LINE_RE = re.compile(r"^\s+at\s+[\w.$<>/ -]+(?:\(|\s)([^()\s]+:\d+:\d+)", re.MULTILINE)
ERROR_RE = re.compile(
    r"(?im)^(?:E\s+)?(?:AssertionError|[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception)|failed|FAIL|ERROR)[:\s].*$"
)
TEST_RE = re.compile(
    r"\b(?:[A-Za-z0-9_./-]+::)?(?:test_[A-Za-z0-9_\[\].:-]+|[A-Za-z0-9_.$-]+\.test\.[A-Za-z0-9_.:-]+|[A-Za-z0-9_.$-]+Spec)\b"
)
COMMAND_RE = re.compile(
    r"(?m)^\s*(?:\$|>)?\s*((?:python|pytest|npm|pnpm|yarn|bun|go|cargo|mvn|gradle|make|cmake|docker|kubectl|terraform|ruff|mypy|tsc|node|git)\b[^\n]*)"
)
CODE_BLOCK_RE = re.compile(r"```(?P<lang>[A-Za-z0-9_+-]*)\n(?P<code>[\s\S]*?)```")
SYMBOL_RE = re.compile(
    r"`([A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)?)`"
    r"|\b(?:class|def|function|func|struct|enum|interface|type)\s+([A-Za-z_][A-Za-z0-9_]*)"
    r"|\b([A-Z][A-Za-z0-9_]{2,}|[a-z_][A-Za-z0-9_]*\([^)]*\))"
)
CONSTRAINT_RE = re.compile(
    r"(?im)^\s*(?:[-*]\s*)?(must|should|do not|don't|never|only|keep|preserve|avoid|require[sd]?|without|no remote|local only)\b.*$"
)
INLINE_CONSTRAINT_RE = re.compile(
    r"(?i)\b(must|should|do not|don't|never|keep|preserve|avoid|require[sd]?|without|no remote|local[- ]only)\b[^.\n]*(?:\.|$)"
)
DESTRUCTIVE_RE = re.compile(
    r"(?i)\b(?:rm\s+-rf|drop\s+database|delete\s+all|wipe|destroy|reset\s+--hard|force\s+push|truncate\s+table)\b"
)
SYMBOL_STOPWORDS = {
    "Add",
    "Build",
    "Fix",
    "Implement",
    "Keep",
    "Must",
    "Only",
    "Please",
    "Refactor",
    "Run",
    "Use",
}


@dataclass
class PromptAnalysis:
    prompt: str
    task_type: str = "unknown"
    explicit_file_paths: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    stack_traces: list[str] = field(default_factory=list)
    error_messages: list[str] = field(default_factory=list)
    failing_tests: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    hard_constraints: list[str] = field(default_factory=list)
    risky_destructive_intent: bool = False
    secrets: list[SecretFinding] = field(default_factory=list)
    code_blocks: list[str] = field(default_factory=list)

    def query_text(self) -> str:
        parts = [
            self.prompt,
            " ".join(self.explicit_file_paths),
            " ".join(self.symbols),
            " ".join(self.failing_tests),
            " ".join(self.error_messages),
        ]
        return "\n".join(part for part in parts if part)


def detect_task_type(prompt: str, failing_tests: list[str], errors: list[str]) -> str:
    lower = prompt.lower()
    if "security" in lower or "vulnerability" in lower or "credential" in lower or "secret" in lower:
        return "security"
    if failing_tests or "failing test" in lower or "pytest" in lower or "test failure" in lower:
        return "test_failure"
    if "refactor" in lower or "cleanup" in lower or "restructure" in lower:
        return "refactor"
    if "explain" in lower or "why" in lower or "how does" in lower:
        return "explanation"
    if any(word in lower for word in ["docker", "kubernetes", "terraform", "ci", "workflow", "deploy", "infra"]):
        return "infra"
    if any(word in lower for word in ["add", "implement", "feature", "support", "build"]):
        return "feature"
    if errors or any(word in lower for word in ["bug", "fix", "crash", "exception", "error", "regression"]):
        return "bug_fix"
    return "unknown"


def _clean_path(path: str) -> str:
    path = path.strip("`'\".,)")
    if ":" in path:
        maybe_path, suffix = path.rsplit(":", 1)
        if suffix.isdigit() or re.match(r"\d+-\d+$", suffix):
            return maybe_path
    return path


def _extract_symbols(prompt: str) -> list[str]:
    symbols: list[str] = []
    for match in SYMBOL_RE.finditer(prompt):
        value = next((group for group in match.groups() if group), "")
        if not value:
            continue
        if value.endswith(")"):
            value = value.split("(", 1)[0]
        if len(value) >= 3 and value not in SYMBOL_STOPWORDS:
            symbols.append(value)
    return dedupe_preserve_order(symbols)


def analyze_prompt(prompt: str) -> PromptAnalysis:
    code_blocks = [m.group("code") for m in CODE_BLOCK_RE.finditer(prompt)]
    paths = dedupe_preserve_order(_clean_path(m.group(0)) for m in PATH_RE.finditer(prompt))
    py_stacks = [m.group(0).strip() for m in PY_STACK_RE.finditer(prompt)]
    js_stacks = [m.group(0).strip() for m in JS_STACK_LINE_RE.finditer(prompt)]
    errors = dedupe_preserve_order(m.group(0).strip() for m in ERROR_RE.finditer(prompt))
    failing_tests = dedupe_preserve_order(m.group(0).strip() for m in TEST_RE.finditer(prompt))
    commands = dedupe_preserve_order(m.group(1).strip() for m in COMMAND_RE.finditer(prompt))
    constraints = dedupe_preserve_order(
        [m.group(0).strip(" -*") for m in CONSTRAINT_RE.finditer(prompt)]
        + [m.group(0).strip(" -*") for m in INLINE_CONSTRAINT_RE.finditer(prompt)]
    )
    secrets = scan_text(prompt)
    task_type = detect_task_type(prompt, failing_tests, errors)
    return PromptAnalysis(
        prompt=prompt,
        task_type=task_type,
        explicit_file_paths=paths,
        symbols=_extract_symbols(prompt),
        stack_traces=dedupe_preserve_order(py_stacks + js_stacks),
        error_messages=errors,
        failing_tests=failing_tests,
        commands=commands,
        hard_constraints=constraints,
        risky_destructive_intent=bool(DESTRUCTIVE_RE.search(prompt)),
        secrets=secrets,
        code_blocks=code_blocks,
    )
