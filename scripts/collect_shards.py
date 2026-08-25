"""Collect Phase 2 shard outputs + local enumeration into one file.

Concatenation with dedup by (instance_id, target); completeness check against
the expected key set before any gate memo cites them (plan Section 11).

Usage: python3 scripts/collect_shards.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

P2 = ROOT / "results" / "phase2"
LOCAL = P2 / "instances.jsonl"
SHARD_GLOB = sorted((P2 / "phase2_shards").glob("shard_*.jsonl")) \
    if (P2 / "phase2_shards").exists() else []


def main():
    merged: dict[str, str] = {}
    sources = [("local", LOCAL)] + [(p.name, p) for p in SHARD_GLOB]
    n_by_source: dict[str, int] = {}
    for name, path in sources:
        if not path.exists():
            n_by_source[name] = 0
            continue
        cnt = 0
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                print(f"[collect] skipping corrupt line in {name}")
                continue
            key = rec["instance_id"] + "|" + json.dumps(rec["target"])
            if key not in merged:
                merged[key] = line
                cnt += 1
        n_by_source[name] = cnt
        print(f"[collect] {name}: {cnt} new rows")

    # completeness vs expected keys
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rp2col", ROOT / "scripts" / "run_phase2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    grid = mod.build_grid(pilot=False)
    expected = set()
    from sheafpatternfusion.enumerate_structures import instantiate, pick_targets
    for j in grid:
        vp = {int(k): tuple(v) for k, v in j["structure"]["var_parents"].items()}
        structure = (vp, tuple(tuple(p) for p in j["structure"]["r_parents"]))
        try:
            inst = instantiate(structure, seed=j["draw_seed"])
        except Exception:
            continue
        for t in pick_targets(inst):
            expected.add(j["iid"] + "|" + json.dumps(list(t)))
    missing = expected - set(merged.keys())
    extra = set(merged.keys()) - expected

    out = P2 / "instances_merged.jsonl"
    with open(out, "w") as f:
        for line in merged.values():
            f.write(line + "\n")
    print(f"[collect] merged {len(merged)} rows -> {out.name}; "
          f"expected {len(expected)}; missing {len(missing)}; extra {len(extra)}")
    if missing:
        sample = sorted(missing)[:10]
        print(f"[collect] missing sample: {sample}")
    (P2 / "COLLECT_REPORT.json").write_text(json.dumps({
        "n_merged": len(merged), "n_expected": len(expected),
        "n_missing": len(missing), "n_extra": len(extra),
        "by_source": n_by_source,
    }, indent=1))
    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
