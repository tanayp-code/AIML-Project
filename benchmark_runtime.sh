#!/usr/bin/env bash
# Compare PyTorch (.pt) and ONNX Runtime (.onnx) resource use for the final models.
#
#   ./benchmark_runtime.sh               # default 300 timed runs per configuration
#   ./benchmark_runtime.sh --runs 1000
#   PYTHON=/path/to/python ./benchmark_runtime.sh
#
# onnx + onnxruntime are installed once into ~/.cache/onnx-bench-libs, NOT into your venv,
# and are looked up after the venv's own packages, so nothing in the venv changes.
# Delete that folder afterwards if you no longer need it.

# Refuse to be sourced: 'set -e' and 'exit' would close the calling terminal
if { [ -n "${BASH_SOURCE:-}" ] && [ "${BASH_SOURCE[0]}" != "$0" ]; } || \
   { [ -n "${ZSH_EVAL_CONTEXT:-}" ] && [[ "$ZSH_EVAL_CONTEXT" == *:file* ]]; }; then
    echo "Run this script, do not source it:  ./benchmark_runtime.sh   (or: bash benchmark_runtime.sh)"
    return 1
fi

set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${PYTHON:-}" ]; then
    if [ -x "../cldrive-unofficial/venv/bin/python" ]; then
        PY="../cldrive-unofficial/venv/bin/python"
    elif [ -x "venv/bin/python" ]; then
        PY="venv/bin/python"
    else
        PY="$(command -v python3 || command -v python)"
    fi
else
    PY="$PYTHON"
fi
LIBS="${ONNX_BENCH_LIBS:-$HOME/.cache/onnx-bench-libs}"
export ONNX_BENCH_LIBS="$LIBS"

if [ ! -x "$PY" ]; then
    echo "Python not found at $PY (set PYTHON=/path/to/python)"; exit 1
fi

if ! "$PY" -c "import sys; sys.path.append('$LIBS'); import onnx, onnxruntime" 2>/dev/null; then
    echo "Installing onnx and onnxruntime into $LIBS (about 50 MB download)..."
    "$PY" -m pip install --progress-bar on --target "$LIBS" onnx onnxruntime
fi

"$PY" benchmark_runtime.py "$@"
