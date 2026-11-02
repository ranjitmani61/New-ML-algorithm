"""Run research/final/final.py as a Kaggle TPU kernel.

    python3 kaggle_final.py push   --slug SLUG [--conditions ...] [--seeds ...] [--steps N] [--per-len N]
    python3 kaggle_final.py status --slug SLUG
    python3 kaggle_final.py fetch  --slug SLUG --out DIR

Kaggle uploads only the kernel's code file, so the runner embeds final.py (base64), checks that
JAX sees the TPU (installing jax[tpu] if not), then runs it and writes /kaggle/working/results.json.
--steps / --per-len are only for the timing probe; the registered final uses final.py's defaults.
Needs KAGGLE_API_TOKEN in the environment.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

RUNNER = r'''
import base64, json, os, subprocess, sys, time
CONFIG = __CONFIG__
SRC = __SRC__
os.makedirs("/kaggle/working", exist_ok=True)
open("/kaggle/working/final.py", "wb").write(base64.b64decode(SRC))

def check():
    r = subprocess.run([sys.executable, "-c", "import jax, optax; print(jax.__version__, jax.devices())"],
                       capture_output=True, text=True)
    out = (r.stdout + r.stderr)[-1500:]
    return r.returncode == 0 and "tpu" in r.stdout.lower(), out

ok, info = check()
print(json.dumps({"env_check": ok, "info": info}), flush=True)
if not ok:
    for spec in (["jax[tpu]", "optax"], ["jax[tpu]==0.4.38", "optax"]):
        t = time.time()
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *spec,
                            "-f", "https://storage.googleapis.com/jax-releases/libtpu_releases.html"],
                           capture_output=True, text=True)
        ok, info = check()
        print(json.dumps({"install": spec, "pip_exit": r.returncode, "seconds": round(time.time() - t),
                          "pip_tail": r.stderr[-800:], "env_check": ok, "info": info}), flush=True)
        if ok:
            break
if not ok:
    print("NO_TPU_JAX", flush=True)
    sys.exit(1)
cmd = [sys.executable, "/kaggle/working/final.py", "--out", "/kaggle/working/results.json",
       "--conditions", *CONFIG["conditions"], "--seeds", *map(str, CONFIG["seeds"])]
if CONFIG.get("steps"):
    cmd += ["--steps", str(CONFIG["steps"])]
if CONFIG.get("per_len"):
    cmd += ["--per-len", str(CONFIG["per_len"])]
print(json.dumps({"cmd": cmd, "final_sha256": CONFIG["final_sha256"]}), flush=True)
r = subprocess.run(cmd)
print(json.dumps({"exit": r.returncode}), flush=True)
'''


def kaggle(*args):
    return subprocess.run(["kaggle", *args], capture_output=True, text=True)


def push(slug, conditions, seeds, steps, per_len, user):
    src = open(os.path.join(HERE, "final.py"), "rb").read()
    cfg = {"conditions": conditions, "seeds": seeds, "steps": steps, "per_len": per_len,
           "final_sha256": hashlib.sha256(src).hexdigest()}
    code = RUNNER.replace("__CONFIG__", json.dumps(cfg)).replace("__SRC__", repr(base64.b64encode(src).decode()))
    d = tempfile.mkdtemp(prefix="final_kernel_")
    open(os.path.join(d, "runner.py"), "w").write(code)
    meta = {"id": f"{user}/{slug}", "title": slug, "code_file": "runner.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": True,
            "machine_shape": "TpuV5E8", "enable_internet": True, "dataset_sources": [], "kernel_sources": [],
            "competition_sources": [], "model_sources": []}
    json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=2)
    r = kaggle("kernels", "push", "-p", d)
    print(r.stdout.strip(), r.stderr.strip()[-1500:])
    print(json.dumps(cfg))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push")
    p.add_argument("--slug", required=True)
    p.add_argument("--conditions", nargs="+", default=["baseline", "sigmoid", "fill", "zeros", "pairs", "oracle"])
    p.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102, 103, 104])
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--per-len", type=int, default=0)
    for name in ("status", "fetch"):
        q = sub.add_parser(name)
        q.add_argument("--slug", required=True)
        if name == "fetch":
            q.add_argument("--out", required=True)
    for q in sub.choices.values():
        q.add_argument("--user", default=os.environ.get("KAGGLE_USERNAME", "epicnexus"))
    a = ap.parse_args()
    if a.cmd == "push":
        push(a.slug, a.conditions, a.seeds, a.steps, a.per_len, a.user)
    elif a.cmd == "status":
        r = kaggle("kernels", "status", f"{a.user}/{a.slug}")
        print(r.stdout.strip(), r.stderr.strip())
    else:
        os.makedirs(a.out, exist_ok=True)
        r = kaggle("kernels", "output", f"{a.user}/{a.slug}", "-p", a.out)
        print(r.stdout.strip()[-2000:], r.stderr.strip()[-2000:])


if __name__ == "__main__":
    main()
