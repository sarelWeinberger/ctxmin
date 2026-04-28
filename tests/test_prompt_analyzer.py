from ctxmin.prompt_analyzer import analyze_prompt


def test_analyzer_extracts_deterministic_signals():
    prompt = """
Fix the failure in src/session.py around `refresh_session`.
Must keep this local only.

$ pytest tests/test_session.py::test_refresh_expired

Traceback (most recent call last):
  File "src/session.py", line 42, in refresh_session
    raise ValueError("expired")
ValueError: expired
"""
    analysis = analyze_prompt(prompt)

    assert analysis.task_type == "test_failure"
    assert "src/session.py" in analysis.explicit_file_paths
    assert "refresh_session" in analysis.symbols
    assert "tests/test_session.py::test_refresh_expired" in analysis.failing_tests
    assert any("ValueError" in error for error in analysis.error_messages)
    assert analysis.commands == ["pytest tests/test_session.py::test_refresh_expired"]
    assert any("local only" in constraint.lower() for constraint in analysis.hard_constraints)
    assert analysis.stack_traces


def test_analyzer_detects_destructive_intent_and_secret():
    analysis = analyze_prompt("Run git reset --hard and use token = 'ghp_abcdefghijklmnopqrstuvwx1234'")

    assert analysis.risky_destructive_intent is True
    assert analysis.secrets
