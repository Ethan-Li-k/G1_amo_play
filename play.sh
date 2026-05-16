#!/bin/bash
# play.sh -- one-line G1 dataset replay
#
# Usage:
#   bash play.sh                                # default: bundled sample episode
#   bash play.sh /path/to/episode.parquet       # custom episode
#
# Env-var overrides:
#   CONDA_ENV       conda env to activate (default: g1_amo_play)
#   PSI_DDS_IFACE   network interface to G1 (default: 192.168.123.22)
#
# Hangs G1 in the rig before running. The script:
#   1. activates the conda env (auto-detects miniconda/anaconda location)
#   2. fixes LD_LIBRARY_PATH for casadi (CXXABI_1.3.15)
#   3. invokes play.py against the chosen parquet

set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="${CONDA_ENV:-g1_amo_play}"
PARQUET="${1:-$REPO_DIR/examples/Pull_the_tray_episode_000000.parquet}"

if [ ! -f "$PARQUET" ]; then
    echo "[play.sh] parquet not found: $PARQUET"
    echo "[play.sh] usage: bash play.sh [/path/to/episode.parquet]"
    exit 1
fi

# ---- activate conda ----
if ! command -v conda >/dev/null 2>&1; then
    # Try common conda installation locations
    for cand in \
        "$HOME/miniconda3" \
        "$HOME/anaconda3" \
        "/opt/conda" \
        "/opt/miniconda3" \
        "/opt/anaconda3" \
        "/home/ubuntu/miniconda3" \
    ; do
        if [ -f "$cand/etc/profile.d/conda.sh" ]; then
            # shellcheck disable=SC1091
            source "$cand/etc/profile.d/conda.sh"
            break
        fi
    done
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "[play.sh] conda not found. Either install Miniconda or:"
    echo "          source /path/to/miniconda3/etc/profile.d/conda.sh"
    echo "          before running this script."
    exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[play.sh] conda env '$ENV_NAME' not found."
    echo "[play.sh] create it first:"
    echo "          conda env create -f $REPO_DIR/environment.yml"
    echo "          (or set CONDA_ENV to an existing env name)"
    exit 1
fi

conda activate "$ENV_NAME"

# ---- libstdc++ fix for casadi 3.7 (see README §troubleshooting) ----
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# ---- DDS network iface ----
export PSI_DDS_IFACE="${PSI_DDS_IFACE:-192.168.123.22}"

cd "$REPO_DIR"
exec python play.py --parquet "$PARQUET"
