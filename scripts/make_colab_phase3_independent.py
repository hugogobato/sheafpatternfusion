"""Generate fully independent continuation notebooks for the remaining scaling work.

Each notebook is self-contained, hardcodes its assigned structures (no
sample_structures call at runtime that could diverge), and writes to a
SLICE-SPECIFIC output file so parallel runs never collide by filename.

- Engine fleet: 33 notebooks, each 2-4 n=5 structures (4-8 engine rows).
  Median wall 0.9-1.5 h, worst 3.4 h (hard wall 4 h).
- Finish fleet: 12 notebooks, each handles the single n=6 pilot + quota-6
  attacks for its shard after the engine fleet lands. Wall <1 h.

All notebooks start by cloning the repo (so they can optionally read the
already-committed partial scaling_probe files), but the engine slices do NOT
depend on that clone to decide what to run - their job list is baked in.

Outputs land in /content/results/phase3/ as:
  scaling_resume_sXX_pY.jsonl   (engine slices, 2-4 iids each)
  scaling_finish_sXX.json / attacks

A local collector merges them:
  python3 scripts/collect_phase3_remaining.py

Usage: python3 scripts/make_colab_phase3_independent.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUTDIR = ROOT / "notebooks_colab" / "phase3b_independent"
CHECKPOINT_DIR = ROOT / "notebooks_colab" / "phase3"
CFG_PATH = ROOT / "configs" / "phase3" / "scaling.json"

LIB_FILES = [
    "mdag_dgp.py",
    "lp_ground_truth.py",
    "enumerate_structures.py",
    "gluing.py",
    "battery.py",
    "engine2.py",
    "attackers.py",
    "phase3_probe.py",
]
REL_IMPORT = re.compile(r"^(\s*)from\s+\.[\w\.]*\s+import[\s(]", re.M)

def build_lib() -> str:
    chunks=[]
    for name in LIB_FILES:
        text=(ROOT/"src"/"sheafpatternfusion"/name).read_text()
        lines=text.splitlines(keepends=True)
        out=[]; skipping=False
        for line in lines:
            if skipping:
                out.append("")
                if ")" in line: skipping=False
                continue
            stripped=line.lstrip()
            if stripped.startswith("from __future__"):
                out.append(""); continue
            if REL_IMPORT.match(line):
                out.append("")
                if "(" in line and ")" not in line: skipping=True
                continue
            if stripped.startswith("import ."):
                out.append(""); continue
            out.append(line)
        chunks.append("".join(out))
    return "# EMBEDDED LIB\nfrom __future__ import annotations\n" + "\n".join(chunks)

def md_cell(text): return {"cell_type":"markdown","metadata":{},"source":text.splitlines(keepends=True)}
def code_cell(lines):
    if isinstance(lines,str): lines=lines.splitlines(keepends=True)
    return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":lines}
def notebook(name,cells): return {"nbformat":4,"nbformat_minor":0,"metadata":{"colab":{"name":name},"kernelspec":{"name":"python3","display_name":"Python 3"}},"cells":cells}
def install_cell():
    return code_cell("""
import importlib.metadata as md, subprocess, sys
WANT={'numpy':'2.4.3','scipy':'1.17.1'}
def _ver(p):
 try: return md.version(p)
 except: return None
missing={p:v for p,v in WANT.items() if _ver(p)!=v}
if not missing:
 print('environment OK:',WANT)
else:
 print('installing pinned numpy/scipy ...')
 res=subprocess.run([sys.executable,'-m','pip','install','-q','numpy==2.4.3','scipy==1.17.1'])
 if res.returncode!=0: raise RuntimeError('pip install failed')
 print('='*72); print('DEPENDENCIES INSTALLED. Runtime > Restart session, then Run all again'); print('='*72); raise SystemExit('restart required')
""".lstrip().splitlines(keepends=True))

ENV_CELL=["import functools\n","import glob\n","import io\n","import json\n","import multiprocessing as mp\n","import os\n","import pathlib\n","import shutil\n","import subprocess\n","import time\n","import urllib.request\n","\n","os.environ['OMP_NUM_THREADS']='1'\n","os.environ['OPENBLAS_NUM_THREADS']='1'\n","os.environ['MKL_NUM_THREADS']='1'\n"]
# Safer: define it inline (copy from original)
RUNNER_HELPERS = r'''
def _san(o):
    import numpy as _np
    if isinstance(o, (_np.floating,)):
        return float(o)
    if isinstance(o, (_np.integer,)):
        return int(o)
    if isinstance(o, (_np.bool_,)):
        return bool(o)
    if isinstance(o, _np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))
def dump_line(obj):
    return json.dumps(obj, default=_san)
def load_done(path, key_fn):
    done=set()
    if path.exists():
        for line in path.read_text().splitlines():
            try: done.add(key_fn(json.loads(line)))
            except: pass
    return done
def pooled_map_deadline(worker_fn, items, n_workers=2, stall_timeout_s=5400, seconds_budget=None, meta=None):
    from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
    items=list(items)
    if meta is None: meta={}
    meta['not_run']=[]; meta['timed_out']=False; meta['completed']=0; meta['done_keys']=set()
    t_start=time.time()
    def left():
        return None if seconds_budget is None else seconds_budget-(time.time()-t_start)
    def jkey(it):
        if isinstance(it, dict): return it.get('iid') or it.get('instance_id') or id(it)
        return str(it)
    if len(items)<=1 or n_workers<=1:
        for it in items:
            if left() is not None and left()<=0:
                meta['timed_out']=True; meta['not_run']=[jkey(x) for x in items[items.index(it):]]; return
            r=worker_fn(it); meta['completed']+=1; meta['done_keys'].add(jkey(it)); yield r
        return
    ex=ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context('fork'))
    fut_item={}; nxt=0
    try:
        while True:
            while nxt<len(items) and len(fut_item)<2*n_workers:
                if left() is not None and left()<=0: meta['timed_out']=True; break
                f=ex.submit(worker_fn, items[nxt]); fut_item[f]=items[nxt]; nxt+=1
            if meta['timed_out']:
                meta['not_run'].extend(jkey(x) for x in items[nxt:]); nxt=len(items)
            if fut_item:
                done_set,_=wait(set(fut_item), timeout=stall_timeout_s, return_when=FIRST_COMPLETED)
                if not done_set: raise RuntimeError(f'pool stalled {stall_timeout_s}s with {len(fut_item)} futures pending')
                for f in done_set:
                    it=fut_item.pop(f); meta['completed']+=1; meta['done_keys'].add(jkey(it)); yield f.result()
            if nxt>=len(items) and not fut_item: break
            if meta['timed_out']:
                still={}
                for f,it in fut_item.items():
                    if not f.cancel(): still[f]=it
                    else: meta['not_run'].append(jkey(it))
                fut_item=still
        ex.shutdown(wait=False, cancel_futures=True)
    except Exception as e:
        print(f'(pool yielded {meta["completed"]}/{len(items)} then {type(e).__name__}; finishing remainder sequentially)',flush=True)
        for proc in (getattr(ex,'_processes',None) or {}).values():
            try: proc.kill()
            except: pass
        ex.shutdown(wait=False, cancel_futures=True)
        for it in items:
            if jkey(it) in meta['done_keys']: continue
            r=worker_fn(it); meta['completed']+=1; meta['done_keys'].add(jkey(it)); yield r
'''

CLONE_CELL = r'''
REPO = pathlib.Path("/content/sheafpatternfusion")
OUT_DIR = pathlib.Path("/content/results/phase3")
OUT_DIR.mkdir(parents=True, exist_ok=True)
if not REPO.exists():
    print("cloning repo for reference configs/checkpoints ...")
    subprocess.run(["git","clone","https://github.com/hugogobato/sheafpatternfusion.git", str(REPO)], check=False)
else:
    subprocess.run(["git","-C", str(REPO), "pull", "--ff-only"], check=False)
# make the partial checkpoints visible in OUT_DIR if they were committed
CLONE_CKPT = REPO / "notebooks_colab" / "phase3"
if CLONE_CKPT.exists():
    for f in CLONE_CKPT.glob("scaling_probe_shard*.jsonl"):
        dst = OUT_DIR / f.name
        if not dst.exists() and f.stat().st_size>0:
            try: shutil.copy(f, dst); print(f"seeded {f.name} from clone ({f.stat().st_size/1024:.0f} KB)")
            except Exception as e: print(f"seed copy failed for {f.name}: {e}")
print("OUT_DIR:", OUT_DIR, "existing:", sorted(p.name for p in OUT_DIR.glob("*.jsonl"))[:5])
'''

def header_md(title, wp, purpose, expected, extra=""):
    return md_cell(f"# SheafPatternFusion Phase 3 (WP3.0 pivot-gate) - {title}\n\nWork package: **{wp}**. {purpose}\n\nRuntime: CPU-only (~2 cores). Expected wall time: **{expected}**. Everything is checkpointed to JSONL and resume-safe: re-running 'Run all' continues where the session stopped.\n\nFirst run: the first cell installs the pinned numpy/scipy and HALTS with a message. Do Runtime > Restart session once (clears the preloaded binaries), then Runtime > Run all again; the install cell detects the pins and skips. The library is embedded in this notebook (generated from sheafpatternfusion source); no package install is needed." + ("\n\n"+extra if extra else ""))

def footer_cell(zip_name_expr="OUT_DIR / 'phase3b_outputs.zip'"):
    # zip_name_expr is a Python expression evaluated in the notebook namespace
    return code_cell(f"""
import zipfile
zip_path = {zip_name_expr}
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for pat in ['*.jsonl','*.json','*.csv']:
        for fp in sorted(OUT_DIR.glob(pat)):
            # skip the zip itself and any Zone.Identifier
            if fp == zip_path or fp.suffix == '.Identifier':
                continue
            z.write(fp, arcname=fp.name)
print(f"Created zip {{zip_path}} ({{zip_path.stat().st_size/1024:.0f}} KB) with {{len([n for n in zipfile.ZipFile(zip_path).namelist()])}} files", flush=True)
try:
    from google.colab import files
    files.download(str(zip_path))
    print('Downloaded:', zip_path)
except Exception as e:
    print('(Not on Colab / download skipped):', e)
""".lstrip().splitlines(keepends=True))

def footer_cell_simple():
    # fallback for notebooks that predefine zip_path
    return footer_cell()

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    # clean old remaining if needed
    cfg=json.loads(CFG_PATH.read_text())
    lib=build_lib()
    try: compile(lib,"<lib>","exec")
    except SyntaxError as e: raise SystemExit(f"lib syntax error: {e}")
    print(f"lib OK {len(lib)} chars")

    # compute missing offline by reading local checkpoint
    from sheafpatternfusion.phase3_probe import sample_structures
    shard_missing={}
    shard_jobs={}  # shard -> list of job dicts for all 18
    shard_max={}
    for shard in range(12):
        p=CHECKPOINT_DIR / f"scaling_probe_shard{shard:02d}.jsonl"
        lines=[json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
        done_ids={r["instance_id"] for r in lines}
        missing=[f"n5_s{shard:02d}_j{i:04d}" for i in range(18) if f"n5_s{shard:02d}_j{i:04d}" not in done_ids]
        shard_missing[shard]=missing
        # generate all 18 jobs offline to embed exact structures
        jobs=[]
        # replicate mk_jobs logic exactly
        raw=sample_structures(5, 18, int(cfg["seeds"]["structure_seed_base_n5"])+shard, f"n5_s{shard:02d}")
        for k,j in enumerate(raw):
            j["iid"]=f"n5_s{shard:02d}_j{k:04d}"
            j["tag"]="n5"
            j["do_attack"]=False
            jobs.append(j)
        shard_jobs[shard]=jobs
        # max for chunk sizing from existing walls
        walls=[]
        by_id=defaultdict(list)
        for r in lines:
            if r["instance_id"].startswith("n5"):
                by_id[r["instance_id"]].append(r)
        for iid, recs in by_id.items():
            s=recs[0].get("wall_struct_s",0)
            s+=sum(r.get("wall_formula_s",0)+r.get("wall_lp_s",0)+r.get("wall_engine_r1_s",0)+r.get("wall_engine_r2_s",0)+r.get("wall_fiber_s",0) for r in recs)
            walls.append(s)
        shard_max[shard]=max(walls) if walls else 2000

    n_resume=0
    for shard in range(12):
        missing=shard_missing[shard]
        if not missing: continue
        chunk=3 if shard_max[shard]>5500 else 4
        # map iid -> job
        job_by_iid={j["iid"]:j for j in shard_jobs[shard]}
        for pi, start in enumerate(range(0, len(missing), chunk)):
            slice_iids=missing[start:start+chunk]
            slice_jobs=[job_by_iid[iid] for iid in slice_iids]
            # embed slice jobs as JSON
            jobs_json=json.dumps(slice_jobs)
            runner = f'''
T_START=time.time()
SLICE_IDX={pi}
SLICE_NAME=f"s{{SHARD_IDX:02d}}_p{{SLICE_IDX}}"
ENG_SLICE_PATH=OUT_DIR / f"scaling_resume_s{{SHARD_IDX:02d}}_p{{SLICE_IDX}}.jsonl"
MAIN_ENG_PATH=OUT_DIR / f"scaling_probe_shard{{SHARD_IDX:02d}}.jsonl"
SOFT=14400.0

def elapsed(): return time.time()-T_START

SLICE_JOBS=json.loads(r\'\'\'{jobs_json}\'\'\')
print(f"[resume shard {{SHARD_IDX:02d}} slice {{SLICE_IDX}}] {{len(SLICE_JOBS)}} n=5 structures -> {{len(SLICE_JOBS)*2}} engine rows; writing to {{ENG_SLICE_PATH.name}}", flush=True)
# resume within slice file itself
done_slice=load_done(ENG_SLICE_PATH, lambda r: r["instance_id"])
pending=[j for j in SLICE_JOBS if j["iid"] not in done_slice]
print(f"  slice file has {{len(done_slice)}} done, {{len(pending)}} to go", flush=True)
if not pending:
    print(f"  slice already complete", flush=True)
else:
    def engine_worker(job):
        return run_scaling_job(job, ENGINE_ONLY_CFG)
    # pilot
    tp0=time.time()
    pilot_recs=engine_worker(dict(pending[0]))
    per=time.time()-tp0
    eta=per*len(pending)/2/3600 if len(pending)>1 else per/3600
    print(f"  self-pilot {{pending[0]['iid']}}: {{per:.0f}}s/job -> ~{{eta:.1f}} h on 2 workers", flush=True)
    with open(ENG_SLICE_PATH,"a") as f:
        for r in pilot_recs: f.write(dump_line(r)+"\\n")
    with open(MAIN_ENG_PATH,"a") as f:
        for r in pilot_recs: f.write(dump_line(r)+"\\n")
    pending=pending[1:]
    meta={{}}
    t0=time.time()
    for recs in pooled_map_deadline(engine_worker, pending, n_workers=2, stall_timeout_s=float(SCALING_CFG["deadlines"]["stall_timeout_s"]), seconds_budget=max(SOFT-elapsed(),60), meta=meta):
        with open(ENG_SLICE_PATH,"a") as fa, open(MAIN_ENG_PATH,"a") as fb:
            for r in recs:
                line=dump_line(r)+"\\n"
                fa.write(line); fb.write(line)
        if meta["completed"]%2==0:
            el=time.time()-t0
            print(f"  [slice {{SLICE_IDX}}] {{meta['completed']}}/{{len(pending)}} {{el/60:.1f}} min", flush=True)
    if meta.get("not_run"):
        print(f"  deadline guard: {{len(meta['not_run'])}} not run: {{meta['not_run'][:4]}}", flush=True)
    print(f"SLICE {{SLICE_IDX}} DONE in {{elapsed()/3600:.2f}} h", flush=True)
# also report overall shard progress after seeding from clone
all_done=load_done(MAIN_ENG_PATH, lambda r: r["instance_id"])
n5_done=len([i for i in all_done if i.startswith(f"n5_s{{SHARD_IDX:02d}}_")])
print(f"shard {{SHARD_IDX:02d}} n5 progress now {{n5_done}}/18 structures (slice file {{len(load_done(ENG_SLICE_PATH, lambda r: r['instance_id']))}}/{{len(SLICE_JOBS)}})", flush=True)
'''
            cells=[
                header_md(f"WP3.0b RESUME n=5 - shard {shard:02d} slice {pi} ({len(slice_jobs)} structures)",
                          "WP3.0b-RESUME (feeds gate G2.6)",
                          f"Independent engine slice for missing n=5 structures on shard {shard:02d}. Hardcodes its {len(slice_jobs)} structures ({', '.join(slice_iids)}) so no sample_structures RNG needed at runtime. Writes to a slice-specific file scaling_resume_s{shard:02d}_p{pi}.jsonl and also appends to the main scaling_probe_shard{shard:02d}.jsonl. Fully independent - no prior checkpoint required; re-running skips done rows in its own slice file.",
                          "<4 h (slice 2-4 structures) - hard wall 4 h, median ~0.9-1.5 h",
                          extra=f"Shard {shard:02d}: {len(missing)} n=5 still missing at generation time; this slice covers {len(slice_jobs)}: {', '.join(slice_iids)}. Original n4 re-timing already complete; n6/attacks deferred to finish notebooks."),
                install_cell(),
                code_cell(ENV_CELL),
                code_cell(lib),
                code_cell([f"SCALING_CFG=json.loads(r'''{json.dumps(cfg)}''')\n", f"SHARD_IDX={shard}\n", f"ENGINE_ONLY_CFG=dict(SCALING_CFG)\n"]),
                code_cell(RUNNER_HELPERS.lstrip("\n").splitlines(keepends=True)),
                code_cell(CLONE_CELL.lstrip("\n").splitlines(keepends=True)),
                code_cell(runner.lstrip("\n").splitlines(keepends=True)),
                footer_cell(f"OUT_DIR / f'scaling_resume_s{shard:02d}_p{pi}.zip'"),
            ]
            nb=notebook(f"nb30_b_resume_n5_s{shard:02d}_p{pi}", cells)
            out=OUTDIR / f"nb30_b_resume_n5_s{shard:02d}_p{pi}.ipynb"
            out.write_text(json.dumps(nb))
            n_resume+=1
            print(f"wrote {out.name} {slice_iids}")

    # Finish fleet - independent, clones and waits for engine completeness
    finish_runner = r'''
T_START=time.time()
ENG_PATH=OUT_DIR / f"scaling_probe_shard{SHARD_IDX:02d}.jsonl"
ATT_PATH=OUT_DIR / f"scaling_attacks_shard{SHARD_IDX:02d}.jsonl"
SUM_PATH=OUT_DIR / f"scaling_summary_shard{SHARD_IDX:02d}.json"
FINISH_SLICE_PATH=OUT_DIR / f"scaling_finish_s{SHARD_IDX:02d}.jsonl"
SOFT=14400.0
N6_GATE=float(SCALING_CFG["deadlines"]["n6_gate_elapsed_s"])
def elapsed(): return time.time()-T_START
def mk_jobs(n_vars, count, seed, tag):
    jobs=sample_structures(n_vars, count, seed, f"{tag}_s{SHARD_IDX:02d}")
    for k,j in enumerate(jobs):
        j["iid"]=f"{tag}_s{SHARD_IDX:02d}_j{k:04d}"
        j["tag"]=tag; j["do_attack"]=False
    return jobs
jobs_n6=mk_jobs(6, int(SCALING_CFG["design"]["n6_pilot_per_shard"]), int(SCALING_CFG["seeds"]["structure_seed_base_n6"])+SHARD_IDX, "n6")
def engine_worker(job):
    return run_scaling_job(job, ENGINE_ONLY_CFG)
# collect engine rows from clone + local OUT_DIR (handles both pre-commit and independent slice files)
eng_rows=[]
# from main shard file
if ENG_PATH.exists():
    for line in ENG_PATH.read_text().splitlines():
        try: eng_rows.append(json.loads(line))
        except: pass
# from independent slice files for same shard
for sf in sorted(OUT_DIR.glob(f"scaling_resume_s{SHARD_IDX:02d}_*.jsonl")):
    for line in sf.read_text().splitlines():
        try: eng_rows.append(json.loads(line))
        except: pass
# from clone's resume slices if not yet copied to OUT_DIR
CLONE_RESUME = REPO / "notebooks_colab" / "phase3b_independent"
if CLONE_RESUME.exists():
    for sf in sorted(CLONE_RESUME.glob(f"scaling_resume_s{SHARD_IDX:02d}_*.jsonl")):
        for line in sf.read_text().splitlines():
            try: eng_rows.append(json.loads(line))
            except: pass
# also from clone's original phase3
CLONE_MAIN = REPO / "notebooks_colab" / "phase3" / f"scaling_probe_shard{SHARD_IDX:02d}.jsonl"
if CLONE_MAIN.exists():
    for line in CLONE_MAIN.read_text().splitlines():
        try: eng_rows.append(json.loads(line))
        except: pass
# deduplicate by instance_id
seen={}
for r in eng_rows:
    seen[r["instance_id"]]=r
# but we need rows per target, so keep all rows distinct by iid+target
# rebuild eng_rows as deduplicated per target
uniq={}
for r in eng_rows:
    k=r["instance_id"]+"|"+json.dumps(r["target"])
    uniq[k]=r
eng_rows=list(uniq.values())
n5_ids={r["instance_id"] for r in eng_rows if r.get("tag")=="n5"}
print(f"[finish shard {SHARD_IDX:02d}] found {len(n5_ids)}/18 n5 structures across all sources", flush=True)
if len(n5_ids) < int(SCALING_CFG["design"]["n5_structures_per_shard"]):
    need=int(SCALING_CFG["design"]["n5_structures_per_shard"])-len(n5_ids)
    print(f"  n5 still incomplete ({need} missing). Run the resume slices for this shard first and re-run this finish notebook; skipping n6/attacks.", flush=True)
else:
    print(f"  n5 complete - proceeding to n6 + attacks", flush=True)
    # ensure main ENG_PATH has the merged rows (so attacks can read from it)
    # append any missing resume rows to MAIN_ENG_PATH
    existing=load_done(ENG_PATH, lambda r: r["instance_id"]+"|"+json.dumps(r["target"]))
    with open(ENG_PATH,"a") as f:
        for r in eng_rows:
            k=r["instance_id"]+"|"+json.dumps(r["target"])
            if k not in existing:
                f.write(dump_line(r)+"\n")
    # trailing n6 arm
    n6_done=load_done(ENG_PATH, lambda r: r["instance_id"])
    n6_pending=[j for j in jobs_n6 if j["iid"] not in n6_done and elapsed()<N6_GATE]
    for j in n6_pending:
        if elapsed()>N6_GATE:
            print("[n6] gate elapsed; stopping", flush=True); break
        try:
            tp0=time.time()
            recs=engine_worker(dict(j))
            with open(ENG_PATH,"a") as f:
                for r in recs: f.write(dump_line(r)+"\n")
            print(f"  [n6] {j['iid']} {time.time()-tp0:.0f}s", flush=True)
        except Exception as e:
            print(f"  [n6] {j['iid']} FAILED {type(e).__name__}: {e}", flush=True)
    # refresh rows after n6
    eng_rows=[]
    for line in ENG_PATH.read_text().splitlines():
        try:
            r=json.loads(line)
            if r.get("tag")=="n5": eng_rows.append(r)
        except: pass
    targets=[r for r in eng_rows if r["gt_recoverable"].startswith("UNDETERMINED") and r["sheaf_recoverable"]=="RECOVERABLE"]
    quota=int(SCALING_CFG["design"]["attack_quota_per_shard"])
    att_done=load_done(ATT_PATH, lambda r: r["instance_id"]+"|"+json.dumps(r["target"]))
    # also consider clone's attacks
    CLONE_ATT=REPO / "notebooks_colab" / "phase3" / f"scaling_attacks_shard{SHARD_IDX:02d}.jsonl"
    if CLONE_ATT.exists():
        for line in CLONE_ATT.read_text().splitlines():
            try:
                r=json.loads(line); att_done.add(r["instance_id"]+"|"+json.dumps(r["target"]))
            except: pass
    import numpy as np
    rng_a=np.random.default_rng(int(SCALING_CFG["seeds"]["attack_srs_seed_base"])+SHARD_IDX)
    keys=sorted(r["instance_id"]+"|"+json.dumps(r["target"]) for r in targets)
    picked=set()
    if len(keys)>quota:
        picks=rng_a.choice(len(keys), size=quota, replace=False)
        picked={keys[int(i)] for i in sorted(picks)}
    else:
        picked=set(keys)
    pending_att=[]
    for k in sorted(picked):
        if k in att_done: continue
        iid,tgt_json=k.split("|",1)
        row=next(r for r in eng_rows if r["instance_id"]==iid and json.dumps(r["target"])==tgt_json)
        pending_att.append({"instance_id":iid,"target":row["target"],"n_vars":row["n_vars"],"var_parents":row["var_parents"],"r_parents":row["r_parents"],"seed":row["seed"],"fixed_cpt":row.get("fixed_cpt"),"mechanism_class":row["mechanism_class"],"poset_shape":row["poset_shape"],"sheaf_recoverable":row["sheaf_recoverable"]})
    skipped=[k for k in keys if k not in picked]
    print(f"[attacks] {len(targets)} undecided-x-REC; quota {quota}; {len(pending_att)} to run; {len(skipped)} skipped by quota", flush=True)
    kills=0
    if pending_att:
        def attack_worker(row):
            return attack_row_fixed(row, SCALING_CFG["attack"])
        meta={}; t0=time.time()
        for rec in pooled_map_deadline(attack_worker, pending_att, n_workers=2, stall_timeout_s=float(SCALING_CFG["deadlines"]["stall_timeout_s"]), seconds_budget=max(SOFT-elapsed(),60), meta=meta):
            rec["skipped_by_quota"]=False
            with open(ATT_PATH,"a") as f: f.write(dump_line(rec)+"\n")
            if rec.get("verdict")=="CONFIRMED_FALSE_RECOVERABLE":
                kills+=1; print(f"  *** CONFIRMED FALSE RECOVERABLE: {rec['instance_id']} ***",flush=True)
            if meta["completed"]%2==0:
                el=time.time()-t0; per=el/max(meta["completed"],1)
                print(f"  [attacks] {meta['completed']}/{len(pending_att)} {el/60:.1f} min ({per:.0f}s/row)",flush=True)
        if meta.get("not_run"): print(f"  [attacks] deadline guard: {len(meta['not_run'])} not run",flush=True)
    with open(ATT_PATH,"a") as f:
        for k in skipped:
            if k in att_done or k in picked: continue
            iid,tgt_json=k.split("|",1)
            f.write(dump_line({"instance_id":iid,"target":json.loads(tgt_json),"verdict":"SKIPPED_QUOTA","skipped_by_quota":True})+"\n")
    # summary
    allrows=[]
    if ENG_PATH.exists():
        for line in ENG_PATH.read_text().splitlines():
            try: allrows.append(json.loads(line))
            except: pass
    def med(xs):
        xs=[x for x in xs if x is not None]
        return float(__import__('numpy').median(xs)) if xs else None
    by_tag={}
    for tag in ("n4t","n5","n6"):
        sub=[r for r in allrows if r.get("tag")==tag]
        if not sub: continue
        dec=[r for r in sub if not r["gt_recoverable"].startswith("UNDETERMINED")]
        und=[r for r in sub if r["gt_recoverable"].startswith("UNDETERMINED")]
        by_tag[tag]={"rows":len(sub),"instances":len({r["instance_id"] for r in sub}),"decidable":len(dec),"decidability_rate":len(dec)/max(len(sub),1),"verdict_counts":{"gt_"+v:sum(1 for r in sub if r["gt_recoverable"]==v) for v in {r["gt_recoverable"] for r in sub}},"sheaf_counts":{"sheaf_"+v:sum(1 for r in sub if r["sheaf_recoverable"]==v) for v in {r["sheaf_recoverable"] for r in sub}},"median_wall_cert_pipeline_s":med([r["wall_struct_s"]+r["wall_formula_s"]+r["wall_lp_s"]+r["wall_engine_r1_s"]+r["wall_engine_r2_s"]+r["wall_fiber_s"] for r in sub]),"median_wall_attack_s":med([r.get("wall_attack_s") or None for r in sub]),"median_frechet_width":med([r.get("frechet_width") for r in sub]),"median_jac_deficiency":med([r.get("jacobian_rank_deficiency") for r in sub]),"undecided_x_sheaf_REC":sum(1 for r in und if r["sheaf_recoverable"]=="RECOVERABLE")}
    attacks=[]
    if ATT_PATH.exists():
        for line in ATT_PATH.read_text().splitlines():
            try:
                a=json.loads(line)
                if a.get("verdict") not in ("SKIPPED_QUOTA",): attacks.append(a)
            except: pass
    summary={"shard":SHARD_IDX,"elapsed_s":round(elapsed(),1),"coverage":{tag:{"rows":v["rows"],"planned_instances":int(SCALING_CFG["design"].get({"n4t":"n4_retime_per_shard","n5":"n5_structures_per_shard","n6":"n6_pilot_per_shard"}[tag]))} for tag,v in by_tag.items()},"by_tag":by_tag,"attacks":{"run":len(attacks),"kills":sum(1 for a in attacks if a.get("verdict")=="CONFIRMED_FALSE_RECOVERABLE"),"median_wall_s":med([a.get("total_wall_s") for a in attacks])},"note":"cert pipeline wall = struct+formula+lp+r1+r2+fiber"}
    SUM_PATH.write_text(json.dumps(summary,indent=1))
    print(json.dumps(summary,indent=1)[:3000])
    print(f"FINISH SHARD {SHARD_IDX} DONE in {elapsed()/3600:.2f} h",flush=True)
'''

    for shard in range(12):
        cells=[
            header_md(f"WP3.0b FINISH - shard {shard:02d} (n6 + attacks + summary)",
                      "WP3.0b-FINISH (feeds gate G2.6)",
                      f"Finalizes shard {shard:02d} after its resume slices complete. Independent - clones the repo and gathers engine rows from all available sources (main file, resume slices, clone). Runs trailing n6 pilot, quota-6 attacks, and rewrites scaling_summary_shard{shard:02d}.json. If n5 still incomplete it exits cleanly.",
                      "<1 h (per shard finish) - hard wall 4 h",
                      extra="Run this AFTER all resume slices for the shard report 18/18 (check via git pull). Re-running is safe."),
            install_cell(),
            code_cell(ENV_CELL),
            code_cell(lib),
            code_cell([f"SCALING_CFG=json.loads(r'''{json.dumps(cfg)}''')\n", f"SHARD_IDX={shard}\n", f"ENGINE_ONLY_CFG=dict(SCALING_CFG)\n"]),
            code_cell(RUNNER_HELPERS.lstrip("\n").splitlines(keepends=True)),
            code_cell(CLONE_CELL.lstrip("\n").splitlines(keepends=True)),
            code_cell(finish_runner.lstrip("\n").splitlines(keepends=True)),
            footer_cell(f"OUT_DIR / f'scaling_finish_s{shard:02d}.zip'"),
        ]
        nb=notebook(f"nb30_b_finish_shard{shard:02d}", cells)
        out=OUTDIR / f"nb30_b_finish_shard{shard:02d}.ipynb"
        out.write_text(json.dumps(nb))
        print(f"wrote {out.name}")

    manifest={
        "generated": __import__('datetime').date.today().isoformat(),
        "fleet":"phase3b_independent (fully independent slices, hardcoded jobs, slice-specific outputs)",
        "total_missing_n5_structures": sum(len(v) for v in shard_missing.values()),
        "resume_notebooks": n_resume,
        "finish_notebooks": 12,
        "total_notebooks": n_resume+12,
        "per_notebook_wall_guarantee":"<4 h (resume 0.9-1.5 h median worst 3.4 h, finish <1 h)",
        "outputs":["/content/results/phase3/scaling_resume_sXX_pY.jsonl (slice engine rows, primary)", "/content/results/phase3/scaling_probe_shardXX.jsonl (mirrored append)", "/content/results/phase3/scaling_attacks_shardXX.jsonl", "/content/results/phase3/scaling_summary_shardXX.json"],
        "usage":"1) Run all resume slices (33, parallel, independent). 2) After each shard's n5 hits 18/18 (check finish notebook's first line), run its finish notebook. 3) Download all *.jsonl/*.json into notebooks_colab/phase3/incoming/ and run python3 scripts/collect_phase3_remaining.py or python3 scripts/run_phase3.py --stage collect."
    }
    (OUTDIR/"MANIFEST_independent.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))
    print(f"\nDone: {OUTDIR} ({n_resume} resume + 12 finish = {n_resume+12} notebooks)")

if __name__=="__main__":
    main()
