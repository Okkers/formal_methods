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

# Marker dir: one empty file per sample that did NOT exit with -1 (255)
MARKER_DIR="$(mktemp -d)"
trap 'rm -rf "$MARKER_DIR"' EXIT

N="$N" EPSILON="$EPSILON" OUTPUT_DIR="$EXP_LOG_DIR" python3 - <<'PY'
import json
import os
from pathlib import Path

output_dir = Path(os.environ["OUTPUT_DIR"])
with (output_dir / "parameters.json").open("w") as parameters_file:
    json.dump(
        {"N": int(os.environ["N"]), "epsilon": float(os.environ["EPSILON"]), "indices": []},
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
        --dataset breast_cancer
    local rc=$?

    # Python's exit(-1) shows up as 255 in the shell
    if [[ $rc -eq 255 ]]; then
        echo "Sample $i exited with -1, not recording index" >&2
    else
        touch "$MARKER_DIR/$i"
        [[ $rc -ne 0 ]] && echo "Sample $i exited with code $rc" >&2
    fi
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

# Write the collected indices into parameters.json
OUTPUT_DIR="$EXP_LOG_DIR" MARKER_DIR="$MARKER_DIR" python3 - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["OUTPUT_DIR"]) / "parameters.json"
indices = sorted(int(p.name) for p in Path(os.environ["MARKER_DIR"]).iterdir())

params = json.loads(path.read_text())
params["indices"] = indices
path.write_text(json.dumps(params, indent=2) + "\n")
PY

echo "Done. Results saved in $EXP_LOG_DIR"