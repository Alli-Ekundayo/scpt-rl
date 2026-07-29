#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
(cd "$REPO_ROOT/rust/pcb_parser" && maturin develop --release)
