#!/usr/bin/env bash
# run_benchmark_tests.sh
# ---------------------------------
# Batch-runs CONCORDIA over the benchmark dataset with
# all combinations of mode (zero-shot / vote),
# six prompt versions, and the optional similarity-hint flag.
# Produces 24 CSV prediction files in example_data/test_results/.
#
# Usage:  bash run_benchmark_tests.sh
# ---------------------------------
set -euo pipefail

DATA="eval/Benchmark_subset__200_pairs_no_label_v1.csv"   # unlabeled benchmark pairs
CFG="config.yaml"                                         # main runtime config
OUT_DIR="example_data/test_results"
mkdir -p "$OUT_DIR"

PROMPTS=("v1.0" "v2.1" "v3.0" "v3.0-CoT" "v3.1" "v3.1-CoT")
MODES=("zero-shot" "vote")
SIMHINT_FLAGS=("" "--sim-hint")   # "" = no sim-hint, second element adds the flag

TOTAL=$(( ${#PROMPTS[@]} * ${#MODES[@]} * ${#SIMHINT_FLAGS[@]} ))
COUNT=0

for MODE in "${MODES[@]}"; do
  for PROMPT in "${PROMPTS[@]}"; do
    for SH in "${SIMHINT_FLAGS[@]}"; do
      ((COUNT++))

      # Build tidy filename components
      if [[ "$MODE" == "zero-shot" ]]; then
        MODE_TAG="zero"
      else
        MODE_TAG="$MODE"   # "vote"
      fi

      HINT_SUFFIX=""
      if [[ "$SH" == "--sim-hint" ]]; then
        HINT_SUFFIX="_simhint"
      fi

      OUTPUT="$OUT_DIR/gpt4o_${PROMPT}_${MODE_TAG}${HINT_SUFFIX}.csv"

      echo "[${COUNT}/${TOTAL}] Running: MODE=${MODE}, PROMPT=${PROMPT}, SIM_HINT=$([[ -z $SH ]] && echo "no" || echo "yes")"

      # shellcheck disable=SC2086   # we need word-splitting for $SH flag
      concord "$DATA" --mode "$MODE" --llm-model gpt4o --prompt-ver "$PROMPT" $SH \
              --output "$OUTPUT" --cfg "$CFG"
    done
  done
done

echo "All benchmark runs finished. Output CSVs are in $OUT_DIR"
