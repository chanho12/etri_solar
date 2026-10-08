#!/bin/sh

# Create a project-local Conda environment and install requirements.txt.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ENV_DIR="$SCRIPT_DIR/.conda-env"

# Prevent the caller's Python configuration and ~/.local packages from leaking in.
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1

cd "$SCRIPT_DIR"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda was not found. Install Miniconda or Anaconda first." >&2
    exit 1
fi

if [ ! -x "$ENV_DIR/bin/python" ]; then
    echo "[1/2] Creating Conda environment: $ENV_DIR"
    conda env create --yes --prefix "$ENV_DIR" --file "$SCRIPT_DIR/environment.yml"
else
    echo "[1/2] Reusing Conda environment: $ENV_DIR"
fi

# Keep user-site isolation enabled whenever this environment is activated.
conda env config vars set --prefix "$ENV_DIR" PYTHONNOUSERSITE=1 >/dev/null

echo "[2/2] Installing dependencies from requirements.txt"
conda run --no-capture-output --prefix "$ENV_DIR" python -m pip install --upgrade pip setuptools wheel
conda run --no-capture-output --prefix "$ENV_DIR" python -m pip install -r "$SCRIPT_DIR/requirements.txt"
conda run --no-capture-output --prefix "$ENV_DIR" python -m pip check
conda run --no-capture-output --prefix "$ENV_DIR" python -c \
    'import catboost, lightgbm, matplotlib, numpy, pandas, pyarrow, scipy, shap, sklearn, torch, xgboost; print("SOLAR dependency imports: OK")'

echo
echo "Conda environment setup completed."
echo "Activate it with:"
echo "  conda activate \"$ENV_DIR\""
echo "Then start training with:"
echo "  ./run_solar.sh"
