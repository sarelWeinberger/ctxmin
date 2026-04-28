from ctxmin.chunking import chunk_text


def test_python_chunker_prefers_functions_and_classes():
    text = """
class Session:
    def refresh_session(self):
        return "ok"

def helper():
    return 1
""".strip()

    chunks = chunk_text("/repo", "src/session.py", text, language="python")

    symbols = {chunk.symbol_name for chunk in chunks}
    assert {"Session", "refresh_session", "helper"} <= symbols
    assert all(chunk.start_line <= chunk.end_line for chunk in chunks)


def test_markdown_chunker_uses_headings():
    text = "# Intro\nhello\n\n## Install\npip install .\n"

    chunks = chunk_text("/repo", "README.md", text, language="markdown")

    assert [chunk.symbol_name for chunk in chunks] == ["Intro", "Install"]
    assert all(chunk.chunk_type == "docs" for chunk in chunks)
