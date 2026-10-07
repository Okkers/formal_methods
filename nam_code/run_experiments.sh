#!/usr/bin/env bash

# Usage: ./run_experiments.sh --n 20 --epsilon 0.1 --output x [--jobs 4]
# Results are saved to ../results/<output>
# --jobs 1 (default) = sequential, --jobs K = run K samples in parallel

set -uo pipefail

N=""
EPSILON=""
OUTPUT=""
JOBS=1

usage() {
    echo "Usage: $0 --n <num_samples> --epsilon <value> --output <dir_name> [--jobs <parallel_jobs>]" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n)       N="$2";       shift 2 ;;
        --epsilon) EPSILON="$2"; shift 2 ;;
        --output)  OUTPUT="$2";  shift 2 ;;
        --jobs)    JOBS="$2";    shift 2 ;;
        *) echo "Unknown argument: $1" >&2; usage ;;
    esac
done

if [[ -z "$N" || -z "$EPSILON" || -z "$OUTPUT" ]]; then
    usage
fi

EXP_LOG_DIR="../results/${OUTPUT}"
mkdir -p "$EXP_LOG_DIR"
N="$N" EPSILON="$EPSILON" OUTPUT_DIR="$EXP_LOG_DIR" python3 - <<'PY'
import json
import os
from pathlib import Path

output_dir = Path(os.environ["OUTPUT_DIR"])
with (output_dir / "parameters.json").open("w") as parameters_file:
    json.dump(
        {"N": int(os.environ["N"]), "epsilon": float(os.environ["EPSILON"])},
        parameters_file,
        indent=2,
    )
    parameters_file.write("\n")
PY

run_one() {
    local i="$1"
    echo "Running sample_index=$i (epsilon=$EPSILON) -> $EXP_LOG_DIR"
    python3 global_minimal_explanation_binary_abductive_parallel.py \
        --sample_index "$i" \
        --epsilon "$EPSILON" \
        --exp_log_dir "$EXP_LOG_DIR" \
    || echo "Sample $i FAILED" >&2
}

if [[ "$JOBS" -le 1 ]]; then
    for ((i = 0; i < N; i++)); do
        run_one "$i"
    done
else
    for ((i = 0; i < N; i++)); do
        run_one "$i" &
        # Throttle: wait for a slot to free up once we hit $JOBS running jobs
        while [[ $(jobs -rp | wc -l) -ge $JOBS ]]; do
            sleep 0.5
        done
    done
    wait
fi

echo "Done. Results saved in $EXP_LOG_DIR"