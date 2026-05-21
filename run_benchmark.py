#!/usr/bin/env python3
"""
Stepwise Arithmetic Benchmark Runner
=====================================
A single script to run and save all experiments for the algo-reasoning benchmark.

Supports:
  - HuggingFace Hub models or local model paths
  - vLLM (fast, GPU-optimized) or HuggingFace Transformers (CPU/GPU)
  - Multi-GPU via tensor parallelism (vLLM)
  - CPU inference (HuggingFace backend)
  - Auto-resumes interrupted runs
  - Generates a summary CSV after all experiments

Usage examples:

  # vLLM + GPU (recommended for speed)
  python run_benchmark.py --model Qwen/Qwen2.5-7B-Instruct --backend vllm

  # HuggingFace Transformers + CPU
  python run_benchmark.py --model Qwen/Qwen2.5-7B-Instruct --backend hf --device cpu

  # Local model path, multi-GPU
  python run_benchmark.py --model ./models/DeepSeek-R1-Distill-Qwen-1.5B --backend vllm --tensor_parallel_size 4

  # Run only specific tasks/ranges
  python run_benchmark.py --model Qwen/Qwen2.5-7B-Instruct --tasks add subtract --ranges range_0_1 range_1_10

  # No reasoning tokens
  python run_benchmark.py --model Qwen/Qwen2.5-7B-Instruct --reasoning false

  # Skip already-completed experiments (default behavior)
  python run_benchmark.py --model Qwen/Qwen2.5-7B-Instruct --resume
"""

import argparse
import json
import os
import re
import sys
import time
from glob import glob
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ──────────────────────────────────────────────
# Division-by-zero / NaN keyword correction
# (equivalent to the post-processing script)
# ──────────────────────────────────────────────
NAN_KEYWORDS = [
    "<answer>NaN</answer>",
    "<answer>undefined</answer>",
    "Division by zero error",
    "division by zero error",
    "<answer>undefined (division by zero)</answer>",
    "This is undefined",
]

# ──────────────────────────────────────────────
# Experiment matrix  (mirrors your bash script)
# ──────────────────────────────────────────────
ALL_RANGES     = ["range_0_1", "range_1_10", "range_10_100"]
ALL_DTYPES     = ["float", "int"]
ALL_LOOKBACKS  = [f"look_back{i}" for i in range(1, 8)]
ALL_TASKS      = ["mixed", "add", "subtract", "multiply", "divide"]


def is_valid_combination(range_: str, dtype: str, lookback: str, task: str) -> bool:
    """Replicate the skip logic from the bash script."""
    if range_ == "range_0_1" and dtype == "int":
        return False
    if lookback != "look_back1" and task != "mixed":
        return False
    return True


def build_experiment_list(ranges, dtypes, lookbacks, tasks):
    combos = []
    for r, d, lb, t in product(ranges, dtypes, lookbacks, tasks):
        if is_valid_combination(r, d, lb, t):
            combos.append((r, d, lb, t))
    return combos


# ──────────────────────────────────────────────
# Prompt loading
# ──────────────────────────────────────────────
def load_prompts(prompt_path: str) -> dict:
    with open(prompt_path, "r") as f:
        return json.load(f)


def build_prompt(prompts: dict, algo: str, x, y, reasoning: bool) -> str:
    mode = "reasoning" if reasoning else "no_reasoning"
    behavior     = prompts["modes"][mode]["behavior"]
    output_rules = prompts["modes"][mode]["output_rules"]
    return prompts["base"].format(
        behavior=behavior,
        output_rules=output_rules,
        algo=algo,
        x=x,
        y=y,
    )


# ──────────────────────────────────────────────
# Answer extraction helpers
# ──────────────────────────────────────────────
EXAMPLE_PLACEHOLDERS = ["FINAL_RESULT"]


def extract_answers(text: str, gt):
    """Return (first_pred, correct, correct_any)."""
    matches = re.findall(r"<answer>(.*?)<\\?/answer>", text, re.DOTALL)

    genuine = []
    first_pred = None

    for match in matches:
        val = match.strip()
        if val in EXAMPLE_PLACEHOLDERS:
            continue
        try:
            numeric = float(val)
        except ValueError:
            continue
        if np.isinf(numeric) or np.isnan(numeric):
            continue
        # Skip dummy 42 answers unless the ground truth is actually 42
        if gt is not None and not np.isnan(float(gt)) and gt != 42 and numeric == 42:
            continue
        genuine.append(numeric)
        if first_pred is None:
            first_pred = numeric

    gt_val = float(gt) if gt is not None else None

    if gt_val is not None and np.isnan(gt_val):
        # Division-by-zero / undefined expected
        correct_any = any(k in text for k in NAN_KEYWORDS)
        correct     = first_pred is None and correct_any  # no numeric pred → good
        # refine: if model said NaN keyword, count it
        correct = correct_any
        return first_pred, correct, correct_any

    correct_any = any(abs(gt_val - v) < 1e-3 for v in genuine) if gt_val is not None else False
    correct     = (first_pred is not None and abs(first_pred - gt_val) < 1e-3) if gt_val is not None else False
    return first_pred, correct, correct_any


# ──────────────────────────────────────────────
# Backend: vLLM
# ──────────────────────────────────────────────
def run_vllm(model_name, prompts_batch, max_tokens, tensor_parallel_size, gpu_memory_utilization, dtype_str):
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        sys.exit("[ERROR] vLLM is not installed. Run: pip install vllm  OR use --backend hf")

    llm = LLM(
        model=model_name,
        dtype=dtype_str,
        max_model_len=32768,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    sp = SamplingParams(temperature=0, max_tokens=min(max_tokens, 32000))
    return llm, sp


def generate_vllm(llm, sp, prompts):
    outputs = llm.generate(prompts, sp)
    return [o.outputs[0].text for o in outputs], \
           [len(o.prompt_token_ids) for o in outputs], \
           [len(o.outputs[0].token_ids) for o in outputs]


# ──────────────────────────────────────────────
# Backend: HuggingFace Transformers
# ──────────────────────────────────────────────
def load_hf_model(model_name, device, torch_dtype_str):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        sys.exit("[ERROR] transformers is not installed. Run: pip install transformers torch")

    import torch
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(torch_dtype_str, torch.float16)

    print(f"  Loading tokenizer from {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print(f"  Loading model (dtype={torch_dtype_str}, device={device}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to("cpu")
    model.eval()
    return tokenizer, model


def generate_hf(tokenizer, model, prompts, max_tokens, device):
    import torch
    texts, in_toks, out_toks = [], [], []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        n_in  = inputs["input_ids"].shape[-1]
        n_out = output.shape[-1] - n_in
        text  = tokenizer.decode(output[0][n_in:], skip_special_tokens=True)
        texts.append(text)
        in_toks.append(n_in)
        out_toks.append(n_out)
    return texts, in_toks, out_toks


# ──────────────────────────────────────────────
# Single experiment runner
# ──────────────────────────────────────────────
def run_experiment(
    *,
    dataset_path: str,
    output_path: str,
    model_name: str,
    backend,          # "vllm" | "hf"
    prompts: dict,
    reasoning: bool,
    batch_size: int,
    # vLLM options
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    vllm_dtype: str = "float16",
    # HF options
    hf_device: str = "cuda",
    hf_dtype: str = "float16",
    max_tokens: int = 32000,
):
    if not os.path.exists(dataset_path):
        print(f"  [SKIP] Dataset not found: {dataset_path}")
        return None

    df = pd.read_json(dataset_path, lines=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    results   = []
    start_idx = 0
    if os.path.exists(output_path):
        done_df   = pd.read_json(output_path, lines=True)
        start_idx = len(done_df)
        results   = done_df.to_dict("records")
        if start_idx >= len(df):
            print(f"  [DONE] Already complete ({start_idx} rows). Skipping.")
            acc = done_df["correct"].mean() if "correct" in done_df else float("nan")
            return acc

    remaining_df = df.iloc[start_idx:].reset_index(drop=True)

    # ── Load model (once per experiment; caller should cache for efficiency) ──
    # We return a generator closure so the caller can reuse the loaded model
    # across experiments via the `engine` parameter (see run_all_experiments).
    raise RuntimeError("Use run_experiment_with_engine() directly.")


# ──────────────────────────────────────────────
# Core loop (model loaded once, reused)
# ──────────────────────────────────────────────
def run_experiment_with_engine(
    *,
    dataset_path: str,
    output_path: str,
    engine,            # (llm, sp) for vllm | (tokenizer, model) for hf
    backend: str,
    prompts: dict,
    reasoning: bool,
    batch_size: int,
    hf_device: str = "cuda",
    max_tokens: int = 32000,
):
    if not os.path.exists(dataset_path):
        print(f"    [SKIP] Dataset not found: {dataset_path}")
        return None

    df = pd.read_json(dataset_path, lines=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    results   = []
    start_idx = 0
    if os.path.exists(output_path):
        try:
            done_df   = pd.read_json(output_path, lines=True)
            start_idx = len(done_df)
            results   = done_df.to_dict("records")
        except Exception:
            pass
        if start_idx >= len(df):
            acc = pd.DataFrame(results)["correct"].mean() if results else float("nan")
            print(f"    [DONE] Already complete. Accuracy={acc:.4f}")
            return acc

    remaining_df = df.iloc[start_idx:].reset_index(drop=True)

    for batch_start in tqdm(range(0, len(remaining_df), batch_size), desc="    batches", leave=False):
        batch_df = remaining_df.iloc[batch_start: batch_start + batch_size]
        batch_prompts = [
            build_prompt(prompts, row["algorithm"], row["x"], row["y"], reasoning)
            for _, row in batch_df.iterrows()
        ]

        if backend == "vllm":
            llm, sp = engine
            texts, in_toks, out_toks = generate_vllm(llm, sp, batch_prompts)
        else:
            tokenizer, model = engine
            texts, in_toks, out_toks = generate_hf(tokenizer, model, batch_prompts, max_tokens, hf_device)

        for (_, row), text, n_in, n_out in zip(batch_df.iterrows(), texts, in_toks, out_toks):
            gt = row["expected_output"]
            first_pred, correct, correct_any = extract_answers(text, gt)

            results.append({
                "id":                   row["id"],
                "num_steps":            row["num_steps"],
                "x":                    row["x"],
                "y":                    row["y"],
                "intermediate_values":  row["intermediate_values"],
                "expected_output":      float(gt) if not pd.isna(gt) else "nan",
                "model_prediction":     first_pred,
                "correct":              correct,
                "correct@any":          correct_any,
                "input_tokens":         n_in,
                "output_tokens":        n_out,
                "total_tokens":         n_in + n_out,
                "predicted_text":       text,
            })

        # Periodic save
        pd.DataFrame(results).to_json(output_path, orient="records", lines=True)

    acc = pd.DataFrame(results)["correct"].mean()
    print(f"    Accuracy = {acc:.4f}  ({len(results)} samples)")
    return acc


# ──────────────────────────────────────────────
# Summary generator
# ──────────────────────────────────────────────
def generate_summary(results_root: str, model_tag: str):
    files = glob(f"{results_root}/{model_tag}/*/*/*/*/*/*.jsonl") + \
            glob(f"{results_root}/{model_tag}/*/*/*/*/*.jsonl")
    rows = []
    for f in files:
        parts = f.replace(results_root + "/", "").split("/")
        # parts: model / range / dtype / lookback / task / results.jsonl
        if len(parts) < 6:
            continue
        _, range_, dtype_, lookback_, task_, _ = parts[:6]
        try:
            df  = pd.read_json(f, lines=True)
            acc = df["correct"].mean()
            acc_any = df["correct@any"].mean() if "correct@any" in df else float("nan")
            rows.append({
                "model": model_tag,
                "range": range_,
                "dtype": dtype_,
                "lookback": lookback_,
                "task": task_,
                "n_samples": len(df),
                "accuracy": round(acc, 4),
                "accuracy@any": round(acc_any, 4),
            })
        except Exception as e:
            print(f"  [WARN] Could not read {f}: {e}")
    if not rows:
        return None
    summary = pd.DataFrame(rows).sort_values(["range", "dtype", "lookback", "task"])
    return summary


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Stepwise Arithmetic Benchmark — single-script runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Model ──
    p.add_argument("--model", required=True,
                   help="HuggingFace model ID (e.g. Qwen/Qwen2.5-7B-Instruct) "
                        "or path to a local model directory (e.g. ./models/mymodel)")

    # ── Backend ──
    p.add_argument("--backend", choices=["vllm", "hf"], default="vllm",
                   help="Inference backend. 'vllm' (fast, GPU-only) or 'hf' (transformers, CPU/GPU). Default: vllm")

    # ── vLLM options ──
    vg = p.add_argument_group("vLLM options (--backend vllm)")
    vg.add_argument("--tensor_parallel_size", type=int, default=1,
                    help="Number of GPUs for tensor parallelism (vLLM only). Default: 1")
    vg.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                    help="Fraction of GPU memory to use (vLLM only). Default: 0.9")
    vg.add_argument("--vllm_dtype", default="float16",
                    choices=["float16", "bfloat16", "float32", "auto"],
                    help="Model dtype for vLLM. Default: float16")

    # ── HF Transformers options ──
    hg = p.add_argument_group("HuggingFace Transformers options (--backend hf)")
    hg.add_argument("--device", default="cuda",
                    help="Device: 'cuda', 'cpu', 'mps', or 'auto'. Default: cuda")
    hg.add_argument("--hf_dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"],
                    help="Torch dtype for HF model. Default: float16")

    # ── Experiment matrix ──
    eg = p.add_argument_group("Experiment matrix")
    eg.add_argument("--ranges", nargs="+", default=ALL_RANGES,
                    choices=ALL_RANGES,
                    help="Number ranges to evaluate. Default: all")
    eg.add_argument("--dtypes", nargs="+", default=ALL_DTYPES,
                    choices=ALL_DTYPES,
                    help="Data types to evaluate. Default: all")
    eg.add_argument("--lookbacks", nargs="+", default=ALL_LOOKBACKS,
                    help="Look-back settings. Default: look_back1..look_back7")
    eg.add_argument("--tasks", nargs="+", default=ALL_TASKS,
                    choices=ALL_TASKS,
                    help="Arithmetic tasks. Default: all")
    eg.add_argument("--reasoning", default="true",
                    help="Whether model generates chain-of-thought tokens. Default: true")

    # ── Paths ──
    pg = p.add_argument_group("Paths")
    pg.add_argument("--dataset_root", default="dataset",
                    help="Root folder containing datasets. Default: dataset")
    pg.add_argument("--results_root", default="results",
                    help="Root folder for output results. Default: results")
    pg.add_argument("--prompt_path", default="prompts/prompts.json",
                    help="Path to prompts JSON file. Default: prompts/prompts.json")

    # ── Runtime ──
    rg = p.add_argument_group("Runtime")
    rg.add_argument("--batch_size", type=int, default=500,
                    help="Batch size for inference. Default: 500")
    rg.add_argument("--max_tokens", type=int, default=32000,
                    help="Maximum tokens to generate. Default: 32000")
    rg.add_argument("--resume", action="store_true", default=True,
                    help="Resume incomplete runs (default: True)")
    rg.add_argument("--summary_only", action="store_true",
                    help="Skip inference; only regenerate the summary CSV.")

    return p.parse_args()


def main():
    args = parse_args()

    reasoning = args.reasoning.strip().lower() in ("true", "1", "yes")
    model_tag = Path(args.model).name  # works for both HF IDs and local paths

    print("=" * 60)
    print("  Stepwise Arithmetic Benchmark")
    print("=" * 60)
    print(f"  Model       : {args.model}  (tag: {model_tag})")
    print(f"  Backend     : {args.backend}")
    print(f"  Reasoning   : {reasoning}")
    print(f"  Ranges      : {args.ranges}")
    print(f"  Dtypes      : {args.dtypes}")
    print(f"  Lookbacks   : {args.lookbacks}")
    print(f"  Tasks       : {args.tasks}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Results dir : {args.results_root}/{model_tag}/")
    print("=" * 60)

    # ── Build experiment list ──
    experiments = build_experiment_list(args.ranges, args.dtypes, args.lookbacks, args.tasks)
    print(f"\n  Total experiments: {len(experiments)}\n")

    if args.summary_only:
        print("--summary_only: skipping inference.\n")
    else:
        # ── Load prompts ──
        if not os.path.exists(args.prompt_path):
            sys.exit(f"[ERROR] Prompt file not found: {args.prompt_path}")
        prompts = load_prompts(args.prompt_path)

        # ── Load model ONCE ──
        print("Loading model (this may take a while)...")
        t0 = time.time()
        if args.backend == "vllm":
            from vllm import LLM, SamplingParams
            llm = LLM(
                model=args.model,
                dtype=args.vllm_dtype,
                max_model_len=32768,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
            sp = SamplingParams(temperature=0, max_tokens=args.max_tokens)
            engine = (llm, sp)
        else:
            tokenizer, model = load_hf_model(args.model, args.device, args.hf_dtype)
            engine = (tokenizer, model)

        print(f"  Model loaded in {time.time() - t0:.1f}s\n")

        # ── Run experiments ──
        all_accs = {}
        for i, (range_, dtype_, lookback_, task_) in enumerate(experiments, 1):
            dataset_path = (
                f"{args.dataset_root}/{range_}/{dtype_}/{lookback_}/"
                f"algo_reasoning_dataset_{task_}.jsonl"
            )
            output_path = (
                f"{args.results_root}/{model_tag}/{range_}/{dtype_}/{lookback_}/"
                f"{task_}/results.jsonl"
            )

            print(f"[{i:3d}/{len(experiments)}] {range_} | {dtype_} | {lookback_} | {task_}")
            acc = run_experiment_with_engine(
                dataset_path=dataset_path,
                output_path=output_path,
                engine=engine,
                backend=args.backend,
                prompts=prompts,
                reasoning=reasoning,
                batch_size=args.batch_size,
                hf_device=args.device if args.backend == "hf" else "cuda",
                max_tokens=args.max_tokens,
            )
            if acc is not None:
                all_accs[f"{range_}|{dtype_}|{lookback_}|{task_}"] = acc

        print(f"\nAll experiments complete. Mean accuracy: {np.mean(list(all_accs.values())):.4f}\n")

    # ── Generate summary ──
    print("Generating summary CSV...")
    summary = generate_summary(args.results_root, model_tag)
    if summary is not None:
        summary_path = f"{args.results_root}/{model_tag}/summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"  Summary saved to: {summary_path}")
        print("\n" + summary.to_string(index=False))
    else:
        print("  No results found to summarize.")


if __name__ == "__main__":
    main()