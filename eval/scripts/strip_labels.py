#!/usr/bin/env python
"""
Create a version of the benchmark CSV without the label column.
This ensures proper CSV parsing, preserving quoted fields.
"""
import pandas as pd

# Read the original CSV with labels
df = pd.read_csv("eval/Benchmark_subset__200_pairs_v1.csv")

# Remove the relationship_label column
df_no_label = df.drop(columns=["relationship_label"])

# Write to CSV, preserving all fields
df_no_label.to_csv("eval/Benchmark_subset__200_pairs_no_label_v1.csv", index=False)

print(f"Created unlabeled version with {len(df_no_label)} pairs")
