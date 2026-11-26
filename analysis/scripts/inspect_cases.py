"""Print trace context for examples of a given model / error type (manual validation)."""
import json, glob, sys, os, re, random
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import trace_parser as tp
pq, model, etype, k = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
d = pd.read_parquet(pq); d = d[(d.model == model) & (d.error_type == etype)]
random.seed(0)
alg = {}
for f in glob.glob("/Volumes/Crucial/reasoning_cdac/dataset/**/*.jsonl", recursive=True):
    for l in open(f):
        r = json.loads(l); alg[r["id"]] = r["algorithm"]
for _, row in d.sample(min(k, len(d)), random_state=0).iterrows():
    root = "/Volumes/Crucial/final_results" if os.path.isdir(f"/Volumes/Crucial/final_results/{row.model}") else "/Volumes/Crucial/Reasoning/reasoning_soket/results"
    f = f"{root}/{row.model}/{row['range']}/{row['dtype']}/look_back{row.look_back}/{row.task}/results.jsonl"
    x = next(json.loads(l) for l in open(f) if row.id in l)
    steps = tp.parse_algorithm(alg[x["id"]]); ref = tp.ref_values(x["intermediate_values"])
    first, last, _ = tp.extract_states(x["predicted_text"] or "", 200)
    s = int(row.first_error_step) - 1 if row.first_error_step == row.first_error_step else 0
    kk, t, a, op, b = steps[s]
    own = lambda v: ref.get(v) if v <= 2 else first.get(v)
    print("=" * 100); print(f"{row.model} {row['range']}/{row['dtype']}/lb{row.look_back}/{row.task} steps={row.num_steps} first_error_step={kk}: S{t} = S{a} {op} S{b}")
    print(f"  model S{t}={first.get(t)}  own-operands S{a}={own(a)} S{b}={own(b)} -> expected {tp.apply(op, own(a), own(b))}  ref={ref.get(t)}")
    txt = tp.normalise(x["predicted_text"] or "")
    m = re.search(rf"\bS{t}\s*(=|≈|:|is)", txt)
    i = m.start() if m else 0
    print("  ...", txt[max(0, i - 300): i + 300].replace("\n", " ⏎ "))
