#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GRAPH_JSON="$REPO_ROOT/test_fixtures/graphs/VA_precincts_fixed_pops.json"

OBJECTIVE='{
  "objective": "gingles_partial",
  "threshold": 0.5,
  "min_pop": "BVAP",
  "total_pop": "VAP"
}'

cd "$REPO_ROOT"

cargo run --release --bin frcw_short_bursts -- \
    --graph-json "$GRAPH_JSON" \
    --n-steps 2000 \
    --tol 0.05 \
    --pop-col TOTPOP \
    --assignment-col CD_16 \
    --rng-seed 20260409 \
    --n-threads 4 \
    --burst-length 25 \
    --objective "$OBJECTIVE"
