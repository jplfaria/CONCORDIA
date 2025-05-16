#!/bin/bash

# Script to run concord commands sequentially
# Each command will only start after the previous one completes

echo "Starting concord test sequence..."

# Zero-shot tests without sim-hint
concord example_data/annotations_test.csv --mode zero-shot --llm-model gpt4o --prompt-ver v1.0 --output example_data/test_results/gpt4o_v1.0_zero.csv
echo "Completed test 1 of 24"

concord example_data/annotations_test.csv --mode zero-shot --llm-model gpt4o --prompt-ver v2.1 --output example_data/test_results/gpt4o_v2.1_zero.csv
echo "Completed test 2 of 24"

concord example_data/annotations_test.csv --mode zero-shot --llm-model gpt4o --prompt-ver v3.0 --output example_data/test_results/gpt4o_v3.0_zero.csv
echo "Completed test 3 of 24"

concord example_data/annotations_test.csv --mode zero-shot --llm-model gpt4o --prompt-ver v3.0-CoT --output example_data/test_results/gpt4o_v3.0-CoT_zero.csv
echo "Completed test 4 of 24"

concord example_data/annotations_test.csv --mode zero-shot --llm-model gpt4o --prompt-ver v3.1 --output example_data/test_results/gpt4o_v3.1_zero.csv
echo "Completed test 5 of 24"

concord example_data/annotations_test.csv --mode zero-shot --llm-model gpt4o --prompt-ver v3.1-CoT --output example_data/test_results/gpt4o_v3.1-CoT_zero.csv
echo "Completed test 6 of 24"

# Zero-shot tests with sim-hint
concord example_data/annotations_test.csv --mode zero-shot --llm-model gpt4o --prompt-ver v1.0 --sim-hint --output example_data/test_results/gpt4o_v1.0_zero_simhint.csv
echo "Completed test 7 of 24"

concord example_data/annotations_test.csv --mode zero-shot --llm-model gpt4o --prompt-ver v2.1 --sim-hint --output example_data/test_results/gpt4o_v2.1_zero_simhint.csv
echo "Completed test 8 of 24"

concord example_data/annotations_test.csv --mode zero-shot --llm-model gpt4o --prompt-ver v3.0 --sim-hint --output example_data/test_results/gpt4o_v3.0_zero_simhint.csv
echo "Completed test 9 of 24"

concord example_data/annotations_test.csv --mode zero-shot --llm-model gpt4o --prompt-ver v3.0-CoT --sim-hint --output example_data/test_results/gpt4o_v3.0-CoT_zero_simhint.csv
echo "Completed test 10 of 24"

concord example_data/annotations_test.csv --mode zero-shot --llm-model gpt4o --prompt-ver v3.1 --sim-hint --output example_data/test_results/gpt4o_v3.1_zero_simhint.csv
echo "Completed test 11 of 24"

concord example_data/annotations_test.csv --mode zero-shot --llm-model gpt4o --prompt-ver v3.1-CoT --sim-hint --output example_data/test_results/gpt4o_v3.1-CoT_zero_simhint.csv
echo "Completed test 12 of 24"

# Vote tests without sim-hint
concord example_data/annotations_test.csv --mode vote --llm-model gpt4o --prompt-ver v1.0 --output example_data/test_results/gpt4o_v1.0_vote.csv
echo "Completed test 13 of 24"

concord example_data/annotations_test.csv --mode vote --llm-model gpt4o --prompt-ver v2.1 --output example_data/test_results/gpt4o_v2.1_vote.csv
echo "Completed test 14 of 24"

concord example_data/annotations_test.csv --mode vote --llm-model gpt4o --prompt-ver v3.0 --output example_data/test_results/gpt4o_v3.0_vote.csv
echo "Completed test 15 of 24"

concord example_data/annotations_test.csv --mode vote --llm-model gpt4o --prompt-ver v3.0-CoT --output example_data/test_results/gpt4o_v3.0-CoT_vote.csv
echo "Completed test 16 of 24"

concord example_data/annotations_test.csv --mode vote --llm-model gpt4o --prompt-ver v3.1 --output example_data/test_results/gpt4o_v3.1_vote.csv
echo "Completed test 17 of 24"

concord example_data/annotations_test.csv --mode vote --llm-model gpt4o --prompt-ver v3.1-CoT --output example_data/test_results/gpt4o_v3.1-CoT_vote.csv
echo "Completed test 18 of 24"

# Vote tests with sim-hint
concord example_data/annotations_test.csv --mode vote --llm-model gpt4o --prompt-ver v1.0 --sim-hint --output example_data/test_results/gpt4o_v1.0_vote_simhint.csv
echo "Completed test 19 of 24"

concord example_data/annotations_test.csv --mode vote --llm-model gpt4o --prompt-ver v2.1 --sim-hint --output example_data/test_results/gpt4o_v2.1_vote_simhint.csv
echo "Completed test 20 of 24"

concord example_data/annotations_test.csv --mode vote --llm-model gpt4o --prompt-ver v3.0 --sim-hint --output example_data/test_results/gpt4o_v3.0_vote_simhint.csv
echo "Completed test 21 of 24"

concord example_data/annotations_test.csv --mode vote --llm-model gpt4o --prompt-ver v3.0-CoT --sim-hint --output example_data/test_results/gpt4o_v3.0-CoT_vote_simhint.csv
echo "Completed test 22 of 24"

concord example_data/annotations_test.csv --mode vote --llm-model gpt4o --prompt-ver v3.1 --sim-hint --output example_data/test_results/gpt4o_v3.1_vote_simhint.csv
echo "Completed test 23 of 24"

concord example_data/annotations_test.csv --mode vote --llm-model gpt4o --prompt-ver v3.1-CoT --sim-hint --output example_data/test_results/gpt4o_v3.1-CoT_vote_simhint.csv
echo "Completed test 24 of 24"

echo "All concord tests completed successfully!"
