import csv

rows = list(csv.DictReader(open("results.csv")))
bad = {("text", "0.5754"), ("concat", "0.5794"), ("cross", "0.5746")}
seen, out = set(), []
for r in rows:
    if (r["model"], r["accuracy"]) in bad:
        continue
    key = (r["model"], r["aspect_mode"], r["seed"], r["accuracy"])
    if key in seen:
        continue
    seen.add(key)
    out.append(r)

with open("results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(out)

for r in out:
    print(r["model"], r["aspect_mode"], r["seed"], r["accuracy"], r["f1"])