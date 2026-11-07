"""Run research/llm/run_llm.py jobs as a Kaggle GPU kernel (T4 x2, one lane per GPU).

    python3 kaggle_llm.py push --slug SLUG --jobs JOBS.json
    python3 kaggle_llm.py status --slug SLUG
    python3 kaggle_llm.py fetch --slug SLUG --out DIR

JOBS.json: {"lanes": [[job, ...], [job, ...]], "common": {run_llm.py args}} where each job is a dict
of run_llm.py arguments (condition, family, seed, steps, ...). Outputs go to
/kaggle/working/out/<lane>_<index>_<condition>_<family>_s<seed>/. The model comes from Kaggle
Models (qwen-lm/qwen2.5/transformers/0.5b-instruct/1); nothing is downloaded from Hugging Face.
Needs KAGGLE_API_TOKEN in the environment.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_SOURCE = "qwen-lm/qwen2.5/transformers/0.5b-instruct/1"

RUNNER = r'''
import base64, glob, json, os, subprocess, sys, time
CONFIG = __CONFIG__
FILES = __FILES__
ROOT = os.environ.get("KAGGLE_ROOT", "/kaggle")  # overridable only for local tests of this runner
WORK = f"{ROOT}/working"
os.makedirs(f"{WORK}/code", exist_ok=True)
for name, b64 in FILES.items():
    open(f"{WORK}/code/{name}", "wb").write(base64.b64decode(b64))

def has(mod):
    return subprocess.run([sys.executable, "-c", f"import {mod}"], capture_output=True).returncode == 0

env_info = {m: has(m) for m in ("torch", "transformers", "peft")}
if not env_info["peft"]:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft"], capture_output=True, text=True)
    env_info["peft_install_exit"] = r.returncode
    env_info["peft"] = has("peft")
PREFLIGHT = """
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from peft import LoraConfig, get_peft_model
m = Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                 num_attention_heads=2, num_key_value_heads=1))
m = get_peft_model(m, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
m(input_ids=torch.tensor([[1, 2, 3]]), labels=torch.tensor([[1, 2, 3]])).loss.backward()
print("PREFLIGHT_OK")
"""

def preflight():
    r = subprocess.run([sys.executable, "-c", PREFLIGHT], capture_output=True, text=True)
    return "PREFLIGHT_OK" in r.stdout, (r.stdout + r.stderr)[-1500:]

ok, info = preflight()
env_info["lora_preflight"] = ok
if not ok and "torchao" in info:
    # PEFT raises (instead of skipping) when an incompatible torchao is installed; we never use torchao
    r = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], capture_output=True, text=True)
    env_info["torchao_uninstall_exit"] = r.returncode
    ok, info = preflight()
    env_info["lora_preflight_after_torchao_removal"] = ok
if not ok:
    env_info["lora_preflight_error"] = info[-800:]
v = subprocess.run([sys.executable, "-c", "import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__, torch.cuda.device_count())"],
                   capture_output=True, text=True)
env_info["versions"] = (v.stdout + v.stderr).strip()[-400:]
hits = [os.path.dirname(p) for p in glob.glob(f"{ROOT}/input/**/config.json", recursive=True)]
hits = [h for h in hits if "0.5b-instruct" in h.lower()] or hits
env_info["model_dir"] = sorted(hits, key=len)[0] if hits else None
print(json.dumps({"env": env_info, "code_sha256": CONFIG["code_sha256"]}), flush=True)
if not env_info["peft"] or not env_info["model_dir"] or not ok:
    print("SETUP_FAILED", flush=True)
    sys.exit(1)

def lane(i, jobs):
    for j, job in enumerate(jobs):
        args = {**CONFIG["common"], **job}
        script = args.pop("script", "run_llm.py")  # run_multi.py for the three-person test
        out = f"{WORK}/out/{i}_{j}_{args['condition']}_{args.get('family', 'additive')}_s{args.get('seed', 0)}"
        cmd = [sys.executable, f"{WORK}/code/{script}", "--model", env_info["model_dir"], "--out", out]
        for k, val in args.items():
            if isinstance(val, list):
                cmd += [f"--{k.replace('_', '-')}", *map(str, val)]
            else:
                cmd += [f"--{k.replace('_', '-')}", str(val)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), PYTHONPATH=f"{WORK}/code")
        t = time.time()
        r = subprocess.run(cmd, env=env, cwd=f"{WORK}/code")
        print(json.dumps({"lane": i, "job": j, "args": args, "exit": r.returncode, "seconds": round(time.time() - t)}), flush=True)

import threading
threads = [threading.Thread(target=lane, args=(i, jobs)) for i, jobs in enumerate(CONFIG["lanes"])]
for t in threads: t.start()
for t in threads: t.join()
print("ALL_DONE", flush=True)
'''


def kaggle(*args):
    return subprocess.run(["kaggle", *args], capture_output=True, text=True)


def push(slug, jobs_path, user):
    cfg = json.load(open(jobs_path))
    files = {n: open(os.path.join(HERE, n), "rb").read() for n in ("problems.py", "run_llm.py", "multi.py", "run_multi.py")}
    cfg["code_sha256"] = {n: hashlib.sha256(b).hexdigest() for n, b in files.items()}
    code = RUNNER.replace("__CONFIG__", json.dumps(cfg)).replace(
        "__FILES__", repr({n: base64.b64encode(b).decode() for n, b in files.items()}))
    d = tempfile.mkdtemp(prefix="llm_kernel_")
    open(os.path.join(d, "runner.py"), "w").write(code)
    meta = {"id": f"{user}/{slug}", "title": slug, "code_file": "runner.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "machine_shape": "NvidiaTeslaT4", "enable_internet": True, "dataset_sources": [],
            "kernel_sources": [], "competition_sources": [], "model_sources": [MODEL_SOURCE]}
    json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=2)
    r = kaggle("kernels", "push", "-p", d)
    print(r.stdout.strip(), r.stderr.strip()[-1500:])
    print(json.dumps({"code_sha256": cfg["code_sha256"], "lanes": cfg["lanes"]}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push")
    p.add_argument("--slug", required=True)
    p.add_argument("--jobs", required=True)
    for name in ("status", "fetch"):
        q = sub.add_parser(name)
        q.add_argument("--slug", required=True)
        if name == "fetch":
            q.add_argument("--out", required=True)
    for q in sub.choices.values():
        q.add_argument("--user", default=os.environ.get("KAGGLE_USERNAME", "epicnexus"))
    a = ap.parse_args()
    if a.cmd == "push":
        push(a.slug, a.jobs, a.user)
    elif a.cmd == "status":
        r = kaggle("kernels", "status", f"{a.user}/{a.slug}")
        print(r.stdout.strip(), r.stderr.strip())
    else:
        os.makedirs(a.out, exist_ok=True)
        r = kaggle("kernels", "output", f"{a.user}/{a.slug}", "-p", a.out)
        print(r.stdout.strip()[-2000:], r.stderr.strip()[-2000:])


if __name__ == "__main__":
    main()
