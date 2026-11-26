"""Draw a random sample of traces per model and evaluate them (parser validation)."""
import json, glob, random, sys, os, pickle
sys.path.insert(0, os.path.dirname(__file__))
from trace_parser import *
random.seed(1)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
alg = {}
for f in glob.glob("/Volumes/Crucial/reasoning_cdac/dataset/*/*/*/*.jsonl") + glob.glob("/Volumes/Crucial/Reasoning/reasoning_soket/dataset/*/*/*/*.jsonl"):
    for l in open(f):
        r = json.loads(l); alg[r["id"]] = r["algorithm"]
sample = []
for root in ["/Volumes/Crucial/final_results", "/Volumes/Crucial/Reasoning/reasoning_soket/results"]:
    for md in sorted(glob.glob(root + "/*")):
        files = glob.glob(md + "/*/*/*/*/results.jsonl")
        for f in random.sample(files, min(len(files), N)):
            lines = open(f).readlines()
            x = json.loads(random.choice(lines))
            ev = evaluate(x["predicted_text"] or "", parse_algorithm(alg[x["id"]]), ref_values(x["intermediate_values"]), x["num_steps"])
            sample.append(dict(model=md.split("/")[-1], file=f, id=x["id"], steps=x["num_steps"], correct=x["correct"], text=x["predicted_text"] or "", **ev))
pickle.dump(sample, open(sys.argv[2], "wb"))
