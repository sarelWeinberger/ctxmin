#!/usr/bin/env bash
set -euo pipefail

export CTXMIN_EMBEDDING_BACKEND="${CTXMIN_EMBEDDING_BACKEND:-hash}"
ctxmin bench-contextbench --mode gold-only --dataset default --limit "${1:-5}" --budget "${CTXMIN_BUDGET:-6000}"
