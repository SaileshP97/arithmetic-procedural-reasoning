"""One pass over all stored generations -> one row per example (no trace text).

Output: analysis/outputs/per_example.parquet
Run:    python3 extract_all.py [--limit-files N] [--workers W]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trace_parser as tp  # noqa: E402

RESULT_ROOTS = [
    "/Volumes/Crucial/final_results",                       # 13 models, full 55K each
    "/Volumes/Crucial/Reasoning/reasoning_soket/results",   # DeepSeek-V3.2, Kimi-K2.5 (partial traces)
]
DATASET_ROOT = "/Volumes/Crucial/reasoning_cdac/dataset"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "per_example.parquet")

_ALG: dict[str, str] = {}


def _init():
    global _ALG
    for f in glob.glob(DATASET_ROOT + "/**/*.jsonl", recursive=True):
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                _ALG[r["id"]] = r["algorithm"]


def infer_steps(ref: dict, n_steps: int, look_back: int):
    """Reconstruct (step, target, a, op, b) from reference values when the procedure text is unavailable."""
    steps = []
    for k in range(1, n_steps + 1):
        t = k + 2
        r = ref.get(t)
        cands = sorted({1, 2, *range(max(1, t - look_back), t)})
        hits = []
        for a in cands:
            for b in cands:
                for op in "+-*/":
                    v = tp.apply(op, ref.get(a), ref.get(b))
                    if v is not None and r is not None and r.is_finite() and abs(v - r) <= abs(r) * tp.Decimal("1e-9") + tp.Decimal("1e-12"):
                        hits.append((a, op, b))
        if not hits:
            return steps, False  # cannot continue reliably
        a, op, b = hits[0]
        steps.append((k, t, a, op, b))
    return steps, True


def process(path: str):
    root = next(r for r in RESULT_ROOTS if path.startswith(r))
    model, rng, dtype, lb, task = path[len(root) + 1:].split("/")[:5]
    look_back = int(lb.replace("look_back", ""))
    rows, bad = [], 0
    with open(path) as fh:
        for line in fh:
            try:
                x = json.loads(line)
            except Exception:
                bad += 1
                continue
            n = x["num_steps"]
            ref = tp.ref_values(x.get("intermediate_values"))
            alg = _ALG.get(x["id"])
            if alg:
                steps, src = tp.parse_algorithm(alg), "dataset"
            else:
                steps, ok = infer_steps(ref, n, look_back)
                src = "inferred" if ok else "inferred_partial"
            targets = [t for _, t, *_ in steps][:n]
            ev = tp.evaluate(x.get("predicted_text") or "", steps, ref, n)
            st = tp.ref_stats(ref, targets)
            pred = x.get("model_prediction")
            rows.append({
                "model": model, "range": rng, "dtype": dtype, "look_back": look_back, "task": task,
                "id": x["id"], "num_steps": n,
                "correct_raw": bool(x.get("correct")), "correct_any_raw": bool(x.get("correct@any")),
                "answered_raw": pred is not None and not isinstance(pred, str),
                "output_tokens": x.get("output_tokens"),
                "trace_chars": len(x.get("predicted_text") or ""),
                "procedure_source": src,
                **ev, **st,
            })
    return rows, bad, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-files", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 1))
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    files = sorted(f for r in RESULT_ROOTS for f in glob.glob(r + "/*/range_*/*/look_back*/*/results.jsonl"))
    if a.limit_files:
        files = files[:: max(1, len(files) // a.limit_files)][: a.limit_files]
    print(f"{len(files)} files, {a.workers} workers", flush=True)
    t0 = time.time()
    allrows, badtotal = [], 0
    with Pool(a.workers, initializer=_init) as pool:
        for i, (rows, bad, path) in enumerate(pool.imap_unordered(process, files), 1):
            allrows.extend(rows)
            badtotal += bad
            if i % 25 == 0 or i == len(files):
                print(f"  {i}/{len(files)} files, {len(allrows)} rows, {time.time() - t0:.0f}s", flush=True)
    import pandas as pd
    df = pd.DataFrame(allrows)
    for c in df.columns:
        if df[c].dtype == object and c not in ("model", "range", "dtype", "task", "id", "procedure_source", "error_type"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(f"saved {len(df)} rows -> {a.out}; unreadable lines skipped: {badtotal}; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
