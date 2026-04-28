from __future__ import annotations

from .packing import PackResult


def _bullet_lines(values: list[str], fallback: str = "- None detected") -> str:
    if not values:
        return fallback
    return "\n".join(f"- {value}" for value in values)


def render_minimized_prompt(pack: PackResult) -> str:
    analysis = pack.analysis
    lines: list[str] = []
    lines.append("TASK")
    lines.append(analysis.prompt.strip())
    lines.append("")
    lines.append("SUCCESS CRITERIA")
    criteria: list[str] = []
    if analysis.failing_tests:
        criteria.append("Fix the listed failing tests without regressing related behavior.")
    if analysis.task_type == "feature":
        criteria.append("Implement the requested behavior within the stated constraints.")
    elif analysis.task_type == "bug_fix":
        criteria.append("Resolve the reported bug and preserve existing behavior.")
    elif analysis.task_type == "explanation":
        criteria.append("Explain using the repository context below.")
    criteria.append("Respect all hard constraints and avoid destructive actions.")
    lines.append(_bullet_lines(criteria))
    lines.append("")
    lines.append("CONSTRAINTS")
    constraints = list(analysis.hard_constraints)
    constraints.append("Use only local repository context. Do not send repo code or prompts to remote services.")
    if analysis.risky_destructive_intent:
        constraints.append("Prompt contains potentially destructive intent; require explicit confirmation before destructive changes.")
    if analysis.secrets:
        constraints.append("Prompt appears to contain credentials/secrets; do not echo full secret values.")
    lines.append(_bullet_lines(constraints))
    lines.append("")
    lines.append("ERRORS / FAILING TESTS")
    errors: list[str] = []
    errors.extend(analysis.error_messages)
    errors.extend(analysis.failing_tests)
    errors.extend(analysis.commands)
    errors.extend(analysis.stack_traces)
    lines.append("\n".join(errors) if errors else "None detected")
    lines.append("")
    lines.append("EXPLICIT FILES / SYMBOLS")
    explicit: list[str] = []
    explicit.extend(f"file: {path}" for path in analysis.explicit_file_paths)
    explicit.extend(f"symbol: {symbol}" for symbol in analysis.symbols)
    lines.append(_bullet_lines(explicit))
    lines.append("")
    lines.append("RELEVANT CONTEXT")
    if not pack.chunks:
        lines.append("No indexed chunks selected. Run `ctxmin index <repo_path>` or check the repository path.")
    for idx, packed in enumerate(pack.chunks, start=1):
        chunk = packed.ranked.chunk
        symbol = chunk.symbol_name or ""
        lines.append(f"{idx}. file: {chunk.file_path}")
        lines.append(f"   lines: {chunk.start_line}-{chunk.end_line}")
        if symbol:
            lines.append(f"   symbol: {symbol}")
        lines.append(f"   score: {packed.ranked.score:.3f}")
        lines.append(f"   reason: {packed.ranked.reason}")
        lines.append("   content:")
        lines.append(f"   ```{chunk.language if chunk.language != 'unknown' else ''}")
        lines.extend(f"   {line}" for line in chunk.text.splitlines())
        lines.append("   ```")
        lines.append("")
    lines.append("PACKING")
    lines.append(f"- estimated_tokens: {pack.estimated_tokens}")
    lines.append(f"- budget: {pack.budget}")
    lines.append(f"- omitted_ranked_chunks: {pack.omitted_chunks}")
    return "\n".join(lines).rstrip() + "\n"


def render_explain(pack: PackResult) -> str:
    analysis = pack.analysis
    lines: list[str] = []
    lines.append("ANALYSIS")
    lines.append(f"- task_type: {analysis.task_type}")
    lines.append(f"- explicit_file_paths: {analysis.explicit_file_paths}")
    lines.append(f"- symbols: {analysis.symbols}")
    lines.append(f"- failing_tests: {analysis.failing_tests}")
    lines.append(f"- error_messages: {analysis.error_messages}")
    lines.append(f"- commands: {analysis.commands}")
    lines.append(f"- hard_constraints: {analysis.hard_constraints}")
    lines.append(f"- risky_destructive_intent: {analysis.risky_destructive_intent}")
    lines.append(f"- secrets_detected: {len(analysis.secrets)}")
    lines.append("")
    lines.append("SELECTED CHUNKS")
    if not pack.chunks:
        lines.append("No chunks selected.")
    for packed in pack.chunks:
        ranked = packed.ranked
        chunk = ranked.chunk
        c = ranked.components
        lines.append(
            f"- {chunk.file_path}:{chunk.start_line}-{chunk.end_line} "
            f"symbol={chunk.symbol_name or '-'} score={ranked.score:.3f}"
        )
        d = c.as_debug_dict()
        lines.append(
            "  components="
            + ", ".join(f"{key}:{value:.2f}" for key, value in d.items())
        )
        lines.append(f"  reason={ranked.reason}")
    lines.append("")
    lines.append("TOKEN SAVINGS")
    lines.append(f"- estimated_original_tokens: {pack.original_estimated_tokens}")
    lines.append(f"- minimized_tokens: {pack.estimated_tokens}")
    lines.append(f"- savings_ratio: {pack.savings_ratio:.1%}")
    return "\n".join(lines) + "\n"
