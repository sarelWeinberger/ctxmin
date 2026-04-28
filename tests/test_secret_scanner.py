from ctxmin.secret_scanner import scan_text


def test_secret_scanner_detects_common_credentials():
    findings = scan_text("OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz123456")

    assert findings
    assert findings[0].kind in {"openai_api_key", "credential_assignment"}
    assert "..." in findings[0].value_preview
