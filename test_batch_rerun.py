#!/usr/bin/env python3
"""
Evaluate LLM on Step-wise Arithmetic Dataset
-------------------------------------------
Usage:
    python evaluate_model.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --dataset algo_reasoning_dataset.jsonl \
        --output results_qwen \
        --batch_size 8

Works with any HuggingFace causal LLM.
"""

import argparse
import os
import re
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

with open("prompts/prompts.json", "r") as f:
    PROMPTS = json.load(f)


# ===============================
# 1. Build Prompt
# ===============================
def build_prompt(algo, x, y, reasoning):

    mode = "reasoning" if reasoning else "no_reasoning"

    behavior = PROMPTS["modes"][mode]["behavior"]
    output_rules = PROMPTS["modes"][mode]["output_rules"]

    return PROMPTS["base"].format(
        behavior=behavior,
        output_rules=output_rules,
        algo=algo,
        x=x,
        y=y
    )


# ===============================
# 2. Evaluate Dataset
# ===============================
def evaluate(model_name, output_path, batch_size, dataset, reasoning=True):

    print("Loading dataset...")
    df = pd.read_json(dataset, lines=True)

    output_path = f"results_rerun/{output_path}.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Results will be saved to {output_path}")

    results = []
    start_idx = 0

    # Resume if file exists
    if os.path.exists(output_path):
        print("Resuming from existing results...")
        done_df = pd.read_json(output_path, lines=True)
        start_idx = len(done_df)
        results = done_df.to_dict("records")

        if start_idx == 1000:
            print("All rows already processed. Exiting.")
            return

    print("Loading model...")
    #tokenizer, model = load_model(model_name)
    llm = LLM(
                model=model_name,
                dtype="float16",
                max_model_len=32768,
                #gpu_memory_utilization=0.9,
            )
    
    max_len = llm.llm_engine.model_config.max_model_len

    sampling_params = SamplingParams(
            temperature=0,
            max_tokens=min(max_len, 32000),
        )

    print(f"Starting from row {start_idx} with batch_size={batch_size}")

    remaining_df = df.iloc[start_idx:].reset_index(drop=True)

    # Iterate in batches
    for batch_start in tqdm(range(0, len(remaining_df), batch_size)):
        batch_df = remaining_df.iloc[batch_start: batch_start + batch_size]

        batch_rows = [
            {"algo": row["algorithm"], "x": row["x"], "y": row["y"]}
            for _, row in batch_df.iterrows()
        ]

        #batch_outputs = ask_model_batch(tokenizer, model, batch_rows)
        prompts = [build_prompt(r["algo"], r["x"], r["y"], reasoning) for r in batch_rows]

        batch_outputs = llm.generate(prompts, sampling_params)

        for (_, row), output in zip(batch_df.iterrows(), batch_outputs):

            text = output.outputs[0].text
            correct_any = False
            gt = float(row["expected_output"])
            
            matches = re.findall(r"<answer>(.*?)<\\?/answer>", text, re.DOTALL)

            first_pred = None
            genuine_answers = []
            example_placeholders = ["FINAL_RESULT"]

            for match in matches:
                val = match.strip()
                is_bad_number = isinstance(val, float) and (np.isinf(val) or np.isnan(val))
                if val in example_placeholders or (not is_bad_number and gt is not None and gt != 42 and val == "42"):
                    continue

                try:
                    numeric = float(val)
                except ValueError:
                    continue  # Skip non-numeric values that aren't placeholders

                genuine_answers.append(numeric)

                if first_pred is None:
                    first_pred = numeric

            for y_any in genuine_answers:

                if abs(gt - y_any) < 1e-3:
                    correct_any = True
                    break

            correct = first_pred is not None and abs(first_pred - gt) < 1e-3

            input_tokens = len(output.prompt_token_ids)
            output_tokens = len(output.outputs[0].token_ids)
            total_tokens = input_tokens + output_tokens

            results.append({
                "id": row["id"],
                "num_steps": row["num_steps"],
                "x": row["x"],
                "y": row["y"],
                "intermediate_values": row["intermediate_values"],
                "expected_output": gt,
                "model_prediction": first_pred,
                "correct": correct,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "predicted_text": text,
                "correct@any": correct_any
            })

        # Save periodically
        save_every = batch_size
        if len(results) % save_every == 0:
            pd.DataFrame(results).to_json(output_path, orient="records", lines=True)
            print(f"Saved {len(results)} rows...")

    # Final save
    pd.DataFrame(results).to_json(output_path, orient="records", lines=True)

    print("Saved final results.")
    print("Accuracy:", pd.DataFrame(results)["correct"].mean())


# ===============================
# 3. Main
# ===============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="results/results")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Number of samples to process in one forward pass")
    parser.add_argument("--dataset", default="dataset/algo_reasoning_dataset.jsonl",
                        help="Path to the dataset file (JSONL format)")
    parser.add_argument("--reasoning", default="True", help="Whether model generate reasoning tokens or not.")

    args = parser.parse_args()

    evaluate(args.model, args.output, batch_size=args.batch_size, dataset=args.dataset)


if __name__ == "__main__":
    main()