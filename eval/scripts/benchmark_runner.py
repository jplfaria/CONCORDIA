#!/usr/bin/env python3
"""
CLI to run benchmarking tests for CONCORDIA.
"""
import argparse
import os
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Run benchmarking tests for CONCORDIA")
    parser.add_argument("--data", required=True, help="Path to input CSV file")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["zero-shot", "vote"],
        help="List of modes (zero-shot, vote)",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["v1.0", "v2.1", "v3.0", "v3.0-CoT", "v3.1", "v3.1-CoT"],
        help="List of prompt versions",
    )
    parser.add_argument(
        "--hint",
        choices=["off", "on", "both"],
        default="off",
        help="Similarity hint option",
    )
    parser.add_argument("--llm-model", default="gpt4o", help="LLM model to use")
    parser.add_argument("--cfg", default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--out-dir", default="eval/results", help="Directory to save output files"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.hint == "off":
        hints = [""]
    elif args.hint == "on":
        hints = ["--sim-hint"]
    else:
        hints = ["", "--sim-hint"]

    total = len(args.modes) * len(args.prompts) * len(hints)
    count = 0

    for mode in args.modes:
        mode_tag = "zero" if mode == "zero-shot" else mode
        for prompt in args.prompts:
            for hint in hints:
                count += 1
                suffix = "_simhint" if hint == "--sim-hint" else ""
                output_file = os.path.join(
                    args.out_dir, f"{args.llm_model}_{prompt}_{mode_tag}{suffix}.csv"
                )
                print(
                    f"[{count}/{total}] Running MODE={mode}, PROMPT={prompt}, SIM_HINT={'yes' if hint else 'no'} -> {output_file}"
                )
                cmd = [
                    "concord",
                    args.data,
                    "--mode",
                    mode,
                    "--llm-model",
                    args.llm_model,
                    "--prompt-ver",
                    prompt,
                ]
                if hint:
                    cmd.append(hint)
                cmd += ["--output", output_file, "--cfg", args.cfg]
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
