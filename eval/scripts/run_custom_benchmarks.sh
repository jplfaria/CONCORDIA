#!/bin/bash

# --- Configuration ---
# Ensure this path is correct for your benchmark dataset
BENCHMARK_DATASET="eval/datasets/Benchmark_subset__24_pairs_v1.csv" 
# Main config file for concord
CONFIG_FILE="config.yaml" 

# Models to test
MODELS_TO_TEST=("gpt4olatest" "gpto1")
# Prompts/Templates to test (space-separated string)
PROMPTS_TO_TEST="v1.0 v3.1" 
# Modes to test (space-separated string)
MODES_TO_TEST="zero-shot" 

# Similarity hint: "--hint off" ensures no --sim-hint flag is passed to concord
SIMILARITY_HINT_ARG="--hint off"

# LLM Batch Size: Pass 20 to the modified benchmark_runner.py
LLM_BATCH_SIZE_ARG="--llm-batch-size 20"

# Output directory: Creates a unique timestamped directory for this run
BASE_OUT_DIR="eval/results/benchmark_run_$(date +%Y%m%d_%H%M%S)"

echo "Starting benchmark run..."
echo "Dataset: $BENCHMARK_DATASET"
echo "Config for concord: $CONFIG_FILE"
echo "Models: ${MODELS_TO_TEST[*]}"
echo "Prompts: $PROMPTS_TO_TEST"
echo "Modes: $MODES_TO_TEST"
echo "Similarity Hint: ${SIMILARITY_HINT_ARG}"
echo "LLM Batch Size: ${LLM_BATCH_SIZE_ARG}"
echo "Output base directory: $BASE_OUT_DIR"
echo "---"

for model_name in "${MODELS_TO_TEST[@]}"; do
  echo "Running benchmarks for model: $model_name"
  
  MODEL_OUT_DIR="${BASE_OUT_DIR}/${model_name}"
  mkdir -p "$MODEL_OUT_DIR"

  runner_cmd="poetry run python eval/scripts/benchmark_runner.py \\
                --data \"$BENCHMARK_DATASET\" \\
                --modes $MODES_TO_TEST \\
                --prompts $PROMPTS_TO_TEST \\
                $SIMILARITY_HINT_ARG \\
                $LLM_BATCH_SIZE_ARG \\
                --llm-model \"$model_name\" \\
                --cfg \"$CONFIG_FILE\" \\
                --out-dir \"$MODEL_OUT_DIR\""
  
  echo "Command:"
  echo "$runner_cmd"
  
  log_filepath="${MODEL_OUT_DIR}/LOG_benchmark_run_${model_name}.txt"
  echo "Log will be written to: $log_filepath"
  echo "---"

  echo "Executing for $model_name..."
  eval "$runner_cmd" > "$log_filepath" 2>&1
  if [ $? -eq 0 ]; then
    echo "Successfully completed benchmarks for: $model_name"
  else
    echo "ERROR during benchmark run for: $model_name. Check log: $log_filepath"
  fi
  echo "---"

done

echo "All benchmark commands generated and executed."
echo "Ensure 'eval/scripts/benchmark_runner.py' has been updated to handle '--llm-batch-size'."
