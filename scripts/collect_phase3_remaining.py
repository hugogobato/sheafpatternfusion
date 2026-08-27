"""Merge independent slice outputs into the canonical shard files.

The independent fleet writes per-slice files:
  scaling_resume_sXX_pY.jsonl  (engine rows for 2-4 n=5 structures)

Historical partials live in notebooks_colab/phase3/scaling_probe_shard*.jsonl.
This script merges all sources, deduplicates by (instance_id, target), and
writes the canonical 12 shard files to results/phase3/ and to
notebooks_colab/phase3/ (so the finish notebooks see the complete set).

Usage:
  python3 scripts/collect_phase3_remaining.py
  # then:
  python3 scripts/run_phase3.py --stage collect --stage gate
"""

from __future__ import annotations
import json, glob
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
SRC_DIRS = [
    ROOT / "notebooks_colab" / "phase3",
    ROOT / "notebooks_colab" / "phase3b_independent",
    ROOT / "notebooks_colab" / "phase3_remaining",
    ROOT / "notebooks_colab" / "phase3" / "incoming",
    ROOT / "results" / "phase3",
]
OUT = ROOT / "results" / "phase3"
OUT.mkdir(parents=True, exist_ok=True)

def load_rows(p: Path):
    rows=[]
    for line in p.read_text().splitlines():
        line=line.strip()
        if not line: continue
        try: rows.append(json.loads(line))
        except: pass
    return rows

# collect all engine rows
by_shard_target = {}
shard_counts = defaultdict(int)
for d in SRC_DIRS:
    if not d.exists(): continue
    for pat in ["scaling_probe_shard*.jsonl", "scaling_resume_s*.jsonl", "scaling_resume_*.jsonl"]:
        for f in d.glob(pat):
            # skip zone identifiers
            if f.suffix==".Identifier": continue
            for r in load_rows(f):
                if "instance_id" not in r or "target" not in r: continue
                # only engine rows (have tag)
                if r.get("tag") not in ("n4t","n5","n6"): continue
                k = r["instance_id"] + "|" + json.dumps(r["target"])
                if k not in by_shard_target:
                    by_shard_target[k]=r
                shard_counts[f.name]+=1

# group by shard index
shard_rows = defaultdict(list)
for r in by_shard_target.values():
    # shard index from instance_id e.g. n5_s03_j0011
    try:
        shard = int(r["instance_id"].split("_s")[1].split("_")[0])
    except: shard = 99
    shard_rows[shard].append(r)

for shard in range(12):
    rows = sorted(shard_rows.get(shard, []), key=lambda r: (r["instance_id"], json.dumps(r["target"])))
    out = OUT / f"scaling_probe_shard{shard:02d}.jsonl"
    # also mirror to notebooks_colab/phase3 for clone seeding
    mirror = ROOT / "notebooks_colab" / "phase3" / f"scaling_probe_shard{shard:02d}.jsonl"
    # keep existing n4t rows if any missing? we already have them via by_shard_target
    with open(out,"w") as f:
        for r in rows:
            f.write(json.dumps(r)+"\n")
    # mirror
    try: 
        with open(mirror,"w") as f:
            for r in rows: f.write(json.dumps(r)+"\n")
    except: pass
    n5 = len({r["instance_id"] for r in rows if r.get("tag")=="n5"})
    print(f"shard {shard:02d}: {len(rows)} rows ({n5}/18 n5 structures) -> {out}")

print(f"\nTotal unique engine rows: {len(by_shard_target)}")
# also merge attacks if any independent finish outputs exist
att_by_key={}
for d in SRC_DIRS:
    if not d.exists(): continue
    for f in d.glob("scaling_attacks_shard*.jsonl"):
        for r in load_rows(f):
            if "instance_id" not in r: continue
            k=r["instance_id"]+"|"+json.dumps(r["target"])
            att_by_key[k]=r
if att_by_key:
    by_shard_att=defaultdict(list)
    for r in att_by_key.values():
        try: shard=int(r["instance_id"].split("_s")[1].split("_")[0])
        except: shard=99
        by_shard_att[shard].append(r)
    for shard, rows in by_shard_att.items():
        if shard>=12: continue
        out=OUT / f"scaling_attacks_shard{shard:02d}.jsonl"
        with open(out,"w") as f:
            for r in sorted(rows, key=lambda x: x["instance_id"]): f.write(json.dumps(r)+"\n")
        print(f"attacks shard {shard:02d}: {len(rows)} rows -> {out}")

print("\nDone. Now run: python3 scripts/run_phase3.py --stage collect --stage gate")
