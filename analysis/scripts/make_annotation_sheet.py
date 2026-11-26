"""Sample traces for manual validation of the Step-k (SFB) parser (Reviewer FCUr).

20 traces per model, stratified over procedure length; writes a CSV to fill in by hand.
"""
import json, os, random
import pandas as pd
random.seed(13)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = pd.read_parquet(os.path.join(HERE, "outputs/per_example.parquet"),
                    columns=["model", "range", "dtype", "look_back", "task", "id", "num_steps", "n_parsed_states"])
p = pd.read_parquet("/Volumes/Crucial/openreview_attachments/extracted/unified_results.parquet",
                    columns=["model", "range", "dtype", "look_back", "task", "id", "predicted_steps", "comparison"])
p["key"] = p.model.str.lower(); p["look_back"] = p.look_back.str.replace("look_back", "").astype(int)
d["key"] = d.model.str.lower()
d = d.merge(p.drop(columns="model"), on=["key", "range", "dtype", "look_back", "task", "id"], how="inner")
rows = []
for m, g in d.groupby("model"):
    picks = []
    for lo, hi in [(5, 15), (25, 45), (55, 75), (85, 95)]:
        s = g[(g.num_steps >= lo) & (g.num_steps <= hi)]
        picks.append(s.sample(min(5, len(s)), random_state=13))
    rows.append(pd.concat(picks))
s = pd.concat(rows).reset_index(drop=True)
texts = []
for _, r in s.iterrows():
    root = "/Volumes/Crucial/final_results" if os.path.isdir(f"/Volumes/Crucial/final_results/{r.model}") else "/Volumes/Crucial/Reasoning/reasoning_soket/results"
    f = f"{root}/{r.model}/{r['range']}/{r['dtype']}/look_back{r.look_back}/{r.task}/results.jsonl"
    with open(f) as fh:
        x = next(json.loads(l) for l in fh if r.id in l)
    texts.append((x.get("predicted_text") or "")[:30000])
s["trace (first 30k chars)"] = texts
s = s.rename(columns={"predicted_steps": "sfb_step_count", "comparison": "sfb_label", "n_parsed_states": "auto_state_count"})
for c in ["HUMAN: steps actually executed (0..num_steps)", "HUMAN: all executed steps marked 'Step k'? (Y/N)",
          "HUMAN: SFB label correct? (Y/N)", "HUMAN: notes"]:
    s[c] = ""
s.insert(0, "sample_no", range(1, len(s) + 1))
s.drop(columns=["key"]).to_csv(os.path.join(HERE, "manual_validation/sfb_manual_validation_sheet.csv"), index=False)
print(len(s), "rows written")
