#!/usr/bin/env bash
# Reproduces every number, table, log and figure in the report.
# Usage (from the repository root):  bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
mkdir -p results/logs
rm -rf results/models            # model cache is rebuilt deterministically from the seed
START=$(date +%s)
{
  echo "run_all started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  python -m src.q1
  python -m src.q2
  python -m src.q3
  python -m src.make_figures
  echo "run_all finished in $(( $(date +%s) - START ))s"
} 2>&1 | tee results/logs/run_all.log
