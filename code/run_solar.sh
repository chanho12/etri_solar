#!/bin/sh

# Run SOLAR through its project-local Conda environment without activation.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ENV_DIR="$SCRIPT_DIR/.conda-env"

unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda was not found. Install Miniconda or Anaconda first." >&2
    exit 1
fi

if [ ! -x "$ENV_DIR/bin/python" ]; then
    echo "ERROR: Conda environment not found: $ENV_DIR" >&2
    echo "Run ./setup_conda.sh first." >&2
    exit 1
fi

cd "$SCRIPT_DIR"
exec conda run --no-capture-output --prefix "$ENV_DIR" \
    python -u "$SCRIPT_DIR/run.py" --no-frozen "$@"
