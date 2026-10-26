#!/usr/bin/env bash
# Train the ablation variants in two parallel lanes (5 torch threads each) after `ripple train gnn`.
set -uo pipefail
cd "$(dirname "$0")/.."
DATASET=${1:-synthetic}
lane() { for v in "$@"; do RIPPLE_THREADS=5 uv run ripple -v train "$v" --dataset "$DATASET"; done; }
lane gnn_norot gnn_noapt gnn_nogru > data/logs/ablations_a.log 2>&1 &
lane gnn_noconn gnn_noweather lgbm_noweather > data/logs/ablations_b.log 2>&1 &
wait
echo "ablations done"
