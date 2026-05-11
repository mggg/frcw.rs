#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GRAPH_JSON="$REPO_ROOT/test_fixtures/graphs/IA_counties.json"

cd "$REPO_ROOT"

cargo run --release --bin frcw -- \
    --graph-json "$GRAPH_JSON" \
    --n-steps 5000 \
    --tol 0.20 \
    --pop-col TOTPOP \
    --assignment-col CD \
    --rng-seed 20260409 \
    --n-threads 4 \
    --batch-size 8 \
    --variant district-pairs-rmst \
    --writer canonical \
    --show-progress
