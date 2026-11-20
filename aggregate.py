import csv
from collections import defaultdict
import numpy as np

rows = []
with open("results_seed42.csv") as f:
    for r in csv.DictReader(f):
        r["seed"] = "42"
        rows.append(r)
with open("results.csv") as f:
    rows.extend(csv.DictReader(f))

g = defaultdict(list)
for r in rows:
    g[(r["model"], r["aspect_mode"])].append(
        [float(r[k]) * 100 for k in ("accuracy", "precision", "recall", "f1")])

print(f"{'model':<8}{'aspect':<10}{'n':<3}{'Acc':<16}{'Prec':<16}{'Rec':<16}{'F1':<16}")
for (m, a), v in sorted(g.items()):
    v = np.array(v)
    mu = v.mean(0)
    sd = v.std(0, ddof=1) if len(v) > 1 else np.zeros(4)
    print(f"{m:<8}{a:<10}{len(v):<3}" + "".join(f"{mu[i]:.2f} ± {sd[i]:.2f}".ljust(16) for i in range(4)))