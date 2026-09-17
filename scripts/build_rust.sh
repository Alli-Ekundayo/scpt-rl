#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Ensure cargo is on PATH if installed in standard cargo locations
if [ -f "$HOME/.cargo/env" ]; then
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
fi

# In clean Kaggle/Colab environments, install rust if not found
if ! command -v cargo &> /dev/null; then
    echo "Rust/Cargo not found. Installing Rust toolchain..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
fi

(cd "$REPO_ROOT/rust/pcb_parser" && maturin develop --release)
(cd "$REPO_ROOT/rust/pcb_router" && maturin develop --release)
