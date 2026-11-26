"""Build compact per-trace views for validating the Step-k (SFB) counter.

For a stratified sample (10 traces per model), list:
  T      expected number of steps
  G      SFB step count used in the paper (predicted_steps)
  markers  distinct k for which a "Step k" marker occurs (k <= T)
  values   variables S_t with a committed numeric value (independent parser)
Traces where markers and values disagree are printed with condensed context for inspection.
"""
import json, os, re, sys, random
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import trace_parser as tp
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
random.seed(7)
d = pd.read_parquet(os.path.join(HERE, "outputs/per_example.parquet"), columns=["model", "range", "dtype", "look_back", "task", "id", "num_steps"])
p = pd.read_parquet("/Volumes/Crucial/openreview_attachments/extracted/unified_results.parquet",
                    columns=["model", "range", "dtype", "look_back", "task", "id", "predicted_steps", "comparison"])
p["key"] = p.model.str.lower(); p["look_back"] = p.look_back.str.replace("look_back", "").astype(int)
d["key"] = d.model.str.lower()
d = d.merge(p.drop(columns="model"), on=["key", "range", "dtype", "look_back", "task", "id"])
sample = pd.concat([g.sample(min(10, len(g)), random_state=7) for _, g in d.groupby("model")]).reset_index(drop=True)
rows = []
for i, r in sample.iterrows():
    root = "/Volumes/Crucial/final_results" if os.path.isdir(f"/Volumes/Crucial/final_results/{r.model}") else "/Volumes/Crucial/Reasoning/reasoning_soket/results"
    f = f"{root}/{r.model}/{r['range']}/{r['dtype']}/look_back{r.look_back}/{r.task}/results.jsonl"
    with open(f) as fh:
        x = next(json.loads(l) for l in fh if r.id in l)
    text = x.get("predicted_text") or ""
    T = int(r.num_steps)
    markers = sorted({int(k) for k in re.findall(r"Step\s*(\d+)", text) if 1 <= int(k) <= T})
    first, _, _ = tp.extract_states(text, T + 2)
    values = sorted(k - 2 for k in first if 3 <= k <= T + 2)
    rows.append(dict(no=i + 1, model=r.model, T=T, G=r.predicted_steps, sfb=r.comparison, n_markers=len(markers),
                     n_values=len(values), chars=len(text), id=r.id, file=f,
                     missing_marker_but_value=[k for k in values if k not in markers][:12],
                     marker_but_no_value=[k for k in markers if k not in values][:12]))
out = pd.DataFrame(rows)
out.to_csv(os.path.join(HERE, "manual_validation/sfb_validation_sample.csv"), index=False)
print(out[["no", "model", "T", "G", "sfb", "n_markers", "n_values", "chars"]].to_string(index=False))
