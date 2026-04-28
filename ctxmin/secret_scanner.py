from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SecretFinding:
    kind: str
    value_preview: str
    start: int
    end: int


SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("aws_access_key_id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b")),
    ("openai_api_key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    (
        "credential_assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|token|secret|password|passwd|pwd)\b\s*[:=]\s*['\"]?([A-Za-z0-9_\-./+=]{12,})"
        ),
    ),
]


def _preview(value: str) -> str:
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def scan_text(text: str) -> list[SecretFinding]:
    findings: list[SecretFinding] = []
    for kind, pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(1) if match.lastindex else match.group(0)
            findings.append(
                SecretFinding(
                    kind=kind,
                    value_preview=_preview(value),
                    start=match.start(),
                    end=match.end(),
                )
            )
    return findings


def contains_secret(text: str) -> bool:
    return bool(scan_text(text))
