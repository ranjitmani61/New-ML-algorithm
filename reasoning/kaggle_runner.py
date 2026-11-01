"""Model runner: generates pools of reasoning attempts on a Kaggle GPU.

For every question it samples N independent chain-of-thought attempts from an
open model and records, per attempt: the full text, generated-token count,
whether it finished, and per-token signals measured on the model's *raw*
(temperature-1) distribution: log-probability of the chosen token and the
"token confidence" -mean(top-20 log-probabilities). Reasoning algorithms
(stopping, voting, compute allocation) are then evaluated offline by replaying
attempts from these pools, so every algorithm sees identical model behaviour.

Local commands (need KAGGLE_API_TOKEN):
  python kaggle_runner.py push  --slug reasoning-pools --sets smoke --samples 4
  python kaggle_runner.py status --slug reasoning-pools
  python kaggle_runner.py fetch  --slug reasoning-pools --out DIR
On Kaggle the same file runs as the kernel (it finds CONFIG embedded at push).
"""

from __future__ import annotations

import csv
import glob
import gzip
import json
import os
import random
import subprocess
import sys
import time

CONFIG = None  # replaced at push time with the run configuration
BENCHMARKS_B64 = None  # benchmarks.py source, embedded at push time

MODEL_SOURCE = "qwen-lm/qwen2.5-math/transformers/1.5b-instruct/1"
DATASETS = ["thedevastator/grade-school-math-8k-q-a", "open-benchmarks/math-500-measuring-mathematical-problem-solving"]
SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."

# Question sets. Selection is deterministic and shared with benchmarks.py.
SETS = {
    "gsm8k_dev": {"file": "main_train.csv", "pick": ("sample", 400, 20261101), "max_new": 640},
    "gsm8k_test": {"file": "main_test.csv", "pick": ("all",), "max_new": 640},
    "math_dev": {"file": "math_500_test.csv", "pick": ("sample", 150, 20261101), "max_new": 1280},
    "math_test": {"file": "math_500_test.csv", "pick": ("complement", 150, 20261101), "max_new": 1280},
    "smoke": {"file": "math_500_test.csv", "pick": ("sample", 12, 7), "max_new": 768},
}


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    return [{"index": i, "question": r[0], "solution": r[1]} for i, r in enumerate(body)], header


def select(name, rows):
    kind = SETS[name]["pick"]
    if kind[0] == "all":
        idx = list(range(len(rows)))
    else:
        chosen = sorted(random.Random(kind[2]).sample(range(len(rows)), kind[1]))
        idx = chosen if kind[0] == "sample" else sorted(set(range(len(rows))) - set(chosen))
    prefix = name.split("_")[0]
    source = SETS[name]["file"].rsplit(".", 1)[0]
    return [{"id": f"{prefix}:{source}:{rows[i]['index']}", "set": name, **rows[i]} for i in idx]


# ---------------------------------------------------------------- Kaggle side


def find_file(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not hits:
        raise FileNotFoundError(name)
    return sorted(hits)[0]


def find_model_dir():
    """The model's mount point varies (case, version folders); locate its config.json."""
    hits = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/config.json", recursive=True)]
    ranked = sorted(hits, key=lambda d: (not ("1.5b-instruct" in d.lower() and "math" in d.lower()), len(d)))
    if not ranked:
        raise FileNotFoundError("no model config.json under /kaggle/input: " + str(os.listdir("/kaggle/input")))
    return ranked[0]


def worker(gpu, shard_path, out_prefix, cfg):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

    items = [json.loads(line) for line in open(shard_path)]
    sys.path.insert(0, "/kaggle/working")
    import benchmarks as bm

    features = open(f"{out_prefix}.features.jsonl", "w")
    model_dir = cfg.get("model_dir") or find_model_dir()
    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16).to("cuda").eval()
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    pad = tok.convert_tokens_to_ids("<|endoftext|>")
    temperature, top_p, k = cfg["temperature"], cfg["top_p"], 20

    class Signals(LogitsProcessor):
        """Records raw-distribution signals, then applies temperature and nucleus sampling."""

        def reset(self):
            self.prev, self.chosen, self.conf = None, [], []

        def __call__(self, input_ids, scores):
            lp = torch.log_softmax(scores.float(), dim=-1)
            if self.prev is not None:
                self.chosen.append(self.prev.gather(1, input_ids[:, -1:]).squeeze(1))
            self.conf.append(-lp.topk(k, dim=-1).values.mean(-1))
            self.prev = lp
            s = scores.float() / temperature
            if top_p < 1.0:
                sorted_s, order = torch.sort(s, descending=True)
                probs = torch.softmax(sorted_s, dim=-1)
                drop = probs.cumsum(-1) - probs > top_p
                sorted_s = sorted_s.masked_fill(drop, float("-inf"))
                s = torch.full_like(s, float("-inf")).scatter(1, order, sorted_s)
            return s

    signals = Signals()
    n = cfg["samples"]
    rows_target = cfg["rows"]
    out_json = gzip.open(f"{out_prefix}.jsonl.gz", "wt")
    conf_all, lp_all, offsets = [], [], [0]
    start, generated = time.time(), 0
    i = 0
    batch_no = 0
    while i < len(items):
        if time.time() - cfg["started"] > cfg["time_budget_s"]:
            print(json.dumps({"gpu": gpu, "stopped": "time budget", "done": i}), flush=True)
            break
        per = max(1, rows_target // n)
        batch = items[i : i + per]
        max_new = max(SETS[b["set"]]["max_new"] for b in batch)
        texts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": b["question"]}],
                                         tokenize=False, add_generation_prompt=True) for b in batch]
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
        torch.manual_seed(cfg["seed"] * 100003 + gpu * 1009 + batch_no)
        signals.reset()
        try:
            with torch.no_grad():
                out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                     repetition_penalty=1.0, num_return_sequences=n, max_new_tokens=max_new,
                                     logits_processor=LogitsProcessorList([signals]), eos_token_id=eos,
                                     pad_token_id=pad, use_cache=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            rows_target = max(n, rows_target // 2)
            print(json.dumps({"gpu": gpu, "oom": True, "rows": rows_target}), flush=True)
            continue
        gen = out[:, enc["input_ids"].shape[1]:]
        steps = gen.shape[1]
        chosen = signals.chosen + [signals.prev.gather(1, gen[:, -1:]).squeeze(1)]
        conf = torch.stack(signals.conf, 1).cpu().numpy()
        lps = torch.stack(chosen, 1).cpu().numpy()
        gen_cpu = gen.cpu().numpy()
        for b_idx, item in enumerate(batch):
            samples = []
            summaries = []
            for j in range(n):
                r = b_idx * n + j
                ids = gen_cpu[r]
                ends = [t for t in range(steps) if ids[t] in eos]
                length = ends[0] + 1 if ends else steps
                finished = bool(ends)
                text = tok.decode(ids[:length], skip_special_tokens=True)
                c = conf[r, :length].astype("float16")
                l = lps[r, :length].astype("float16")
                conf_all.append(c)
                lp_all.append(l)
                offsets.append(offsets[-1] + length)
                generated += length
                summaries.append(bm.summarize_attempt(text, length, finished, conf[r, :length], lps[r, :length]))
                samples.append({"text": text, "tokens": int(length), "finished": finished,
                                "sum_logprob": float(lps[r, :length].sum()),
                                "mean_conf": float(conf[r, :length].mean()), "array": len(offsets) - 2})
            out_json.write(json.dumps({**{k2: item[k2] for k2 in ("id", "set", "question", "solution")},
                                       "samples": samples, "prompt_tokens": int(enc["attention_mask"][b_idx].sum())}) + "\n")
            features.write(json.dumps({"id": item["id"], "set": item["set"], "gold": bm.gold_answer(item),
                                       "samples": summaries}) + "\n")
        i += len(batch)
        batch_no += 1
        elapsed = time.time() - start
        print(json.dumps({"gpu": gpu, "done": i, "of": len(items), "tokens": generated,
                          "tok_per_s": round(generated / elapsed, 1), "batch_steps": steps}), flush=True)
    out_json.close()
    features.close()
    np.savez_compressed(f"{out_prefix}.npz", conf=np.concatenate(conf_all) if conf_all else np.zeros(0, "float16"),
                        logprob=np.concatenate(lp_all) if lp_all else np.zeros(0, "float16"),
                        offsets=np.array(offsets, dtype=np.int64))
    meta = {"gpu": gpu, "items": len(items), "completed": i, "tokens": generated,
            "seconds": round(time.time() - start, 1), "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0)}
    import transformers
    meta["transformers"] = transformers.__version__
    with open(f"{out_prefix}.meta.json", "w") as fh:
        json.dump(meta, fh)


def kernel_main(cfg):
    import torch

    cfg["started"] = time.time()
    os.makedirs("/kaggle/working/pools", exist_ok=True)
    import base64

    with open("/kaggle/working/benchmarks.py", "wb") as fh:
        fh.write(base64.b64decode(BENCHMARKS_B64))
    cfg["model_dir"] = find_model_dir()
    print(json.dumps({"model_dir": cfg["model_dir"], "inputs": sorted(os.listdir("/kaggle/input")),
                      "gpus": torch.cuda.device_count()}), flush=True)
    items = []
    for name in cfg["sets"]:
        rows, _ = read_rows(find_file(SETS[name]["file"]))
        items.extend(select(name, rows))
    gpus = max(1, torch.cuda.device_count())
    if "__file__" not in globals() or not os.path.exists(__file__):
        gpus = 1  # no file to re-launch per GPU: run in this process on one GPU
    procs = []
    for g in range(gpus):
        shard = f"/kaggle/working/shard{g}.jsonl"
        with open(shard, "w") as fh:
            for item in items[g::gpus]:
                fh.write(json.dumps(item) + "\n")
        if gpus == 1 and ("__file__" not in globals() or not os.path.exists(__file__)):
            worker(0, shard, "/kaggle/working/pools/part0", cfg)
            continue
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}
        err = open(f"/kaggle/working/worker{g}.err", "w")
        procs.append(subprocess.Popen([sys.executable, __file__, "--worker", str(g), shard,
                                       f"/kaggle/working/pools/part{g}", json.dumps(cfg)], env=env, stderr=err))
    codes = [p.wait() for p in procs]
    with open("/kaggle/working/pools/run.json", "w") as fh:
        json.dump({"config": cfg, "worker_exit_codes": codes, "gpus": gpus, "questions": len(items),
                   "model": MODEL_SOURCE, "system_prompt": SYSTEM, "seconds": round(time.time() - cfg["started"], 1)}, fh)
    for g in range(gpus):
        os.remove(f"/kaggle/working/shard{g}.jsonl")
    print("exit codes", codes, flush=True)
    # Fail loudly: Kaggle reports a raised error (with this message) through its API,
    # which is the only feedback available when outputs cannot be downloaded.
    problems = []
    for g in range(gpus):
        path = f"/kaggle/working/pools/part{g}.jsonl.gz"
        lines = sum(1 for _ in gzip.open(path, "rt")) if os.path.exists(path) else 0
        if (codes and codes[g] != 0) or lines == 0 or not os.path.exists(f"/kaggle/working/pools/part{g}.meta.json"):
            tail = ""
            if os.path.exists(f"/kaggle/working/worker{g}.err"):
                tail = open(f"/kaggle/working/worker{g}.err").read()[-1500:]
            problems.append(f"worker {g}: exit {codes[g] if codes else 'n/a'}, {lines} questions written; {tail}")
    if problems:
        raise RuntimeError(" | ".join(problems))
    emit_features(gpus)


def emit_features(gpus, chunk=3000):
    """Print all feature records into the log as checksummed gzip+base64 chunks.

    The log is returned inline by the Kaggle API, so results arrive even when
    output-file downloads are blocked. Reassemble with `kaggle_runner.py fetch-log`.
    """
    import base64
    import hashlib

    blob = b"".join(open(f"/kaggle/working/pools/part{g}.features.jsonl", "rb").read() for g in range(gpus))
    packed = base64.b64encode(gzip.compress(blob, 9)).decode()
    total = (len(packed) + chunk - 1) // chunk
    for i in range(total):
        piece = packed[i * chunk:(i + 1) * chunk]
        print(f"POOLCHUNK {i} {total} {hashlib.sha256(piece.encode()).hexdigest()[:16]} {piece}", flush=True)
    print(f"POOLDONE {hashlib.sha256(packed.encode()).hexdigest()} {len(blob)}", flush=True)


# ---------------------------------------------------------------- local side


def session_log(user, slug):
    """Kernel log and output-file names straight from the Kaggle API (no file download needed)."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as client:
        req = ApiListKernelSessionOutputRequest()
        req.user_name, req.kernel_slug = user, slug
        resp = client.kernels.kernels_api_client.list_kernel_session_output(req)
    lines = ["files: " + ", ".join(f.file_name for f in resp.files)]
    try:
        for entry in json.loads(resp.log or "[]"):
            lines.append(entry.get("data", "").rstrip("\n"))
    except ValueError:
        lines.append(resp.log or "")
    return "\n".join(lines)


def fetch_log(user, slug, out):
    """Reassemble the POOLCHUNK records from the kernel log into a verified features.jsonl.gz."""
    import base64
    import hashlib

    chunks, done = {}, None
    for line in session_log(user, slug).splitlines():
        if line.startswith("POOLCHUNK "):
            _, i, total, digest, piece = line.split(" ", 4)
            if hashlib.sha256(piece.encode()).hexdigest()[:16] != digest:
                raise ValueError(f"chunk {i} failed its checksum")
            chunks[int(i)] = (int(total), piece)
        elif line.startswith("POOLDONE "):
            done = line.split()
    if not chunks or done is None:
        raise ValueError("no complete feature transport in the log")
    total = next(iter(chunks.values()))[0]
    missing = [i for i in range(total) if i not in chunks]
    if missing:
        raise ValueError(f"{len(missing)} of {total} chunks missing (log truncated?)")
    packed = "".join(chunks[i][1] for i in range(total))
    if hashlib.sha256(packed.encode()).hexdigest() != done[1]:
        raise ValueError("reassembled transport failed its checksum")
    blob = gzip.decompress(base64.b64decode(packed))
    if len(blob) != int(done[2]):
        raise ValueError("size mismatch")
    with gzip.open(out, "wb") as fh:
        fh.write(blob)
    return {"chunks": total, "bytes": len(blob), "questions": blob.count(b"\n"), "out": out,
            "sha256": hashlib.sha256(blob).hexdigest()}


def kaggle(*args):
    return subprocess.run(["kaggle", *args], capture_output=True, text=True)


def push(slug, sets, samples, rows, temperature, top_p, seed, budget_hours, user):
    import tempfile

    cfg = {"sets": sets, "samples": samples, "rows": rows, "temperature": temperature, "top_p": top_p,
           "seed": seed, "time_budget_s": int(budget_hours * 3600)}
    for s in sets:
        if s not in SETS:
            raise SystemExit(f"unknown set {s}")
    import base64

    bench = base64.b64encode(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks.py"), "rb").read()).decode()
    src = open(__file__, encoding="utf-8").read()
    src = src.replace("CONFIG = None  # replaced at push time", f"CONFIG = {json.dumps(cfg)}  # embedded at push time", 1)
    src = src.replace("BENCHMARKS_B64 = None  # benchmarks.py source", f"BENCHMARKS_B64 = {bench!r}  # benchmarks.py source", 1)
    d = tempfile.mkdtemp(prefix="kernel_")
    with open(os.path.join(d, "runner.py"), "w", encoding="utf-8") as fh:
        fh.write(src)
    meta = {"id": f"{user}/{slug}", "title": slug, "code_file": "runner.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": False, "dataset_sources": DATASETS, "kernel_sources": [],
            "competition_sources": [], "model_sources": [MODEL_SOURCE], "machine_shape": "NvidiaTeslaT4"}
    with open(os.path.join(d, "kernel-metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    r = kaggle("kernels", "push", "-p", d)
    print(r.stdout, r.stderr)
    return cfg


def main(argv):
    if len(argv) >= 2 and argv[1] == "--worker":
        worker(int(argv[2]), argv[3], argv[4], json.loads(argv[5]))
        return 0
    if CONFIG is not None:
        kernel_main(dict(CONFIG))
        return 0
    import argparse

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push")
    p.add_argument("--slug", required=True)
    p.add_argument("--sets", nargs="+", required=True)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--rows", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--budget-hours", type=float, default=11.0)
    p.add_argument("--user", default=os.environ.get("KAGGLE_USERNAME", "epicnexus"))
    for name in ("status", "fetch", "log", "fetch-log"):
        q = sub.add_parser(name)
        q.add_argument("--slug", required=True)
        q.add_argument("--user", default=os.environ.get("KAGGLE_USERNAME", "epicnexus"))
        if name in ("fetch", "fetch-log"):
            q.add_argument("--out", required=True)
    args = ap.parse_args(argv[1:])
    if args.cmd == "push":
        print(json.dumps(push(args.slug, args.sets, args.samples, args.rows, args.temperature, args.top_p,
                              args.seed, args.budget_hours, args.user)))
    elif args.cmd == "log":
        print(session_log(args.user, args.slug))
    elif args.cmd == "fetch-log":
        print(json.dumps(fetch_log(args.user, args.slug, args.out)))
    elif args.cmd == "status":
        r = kaggle("kernels", "status", f"{args.user}/{args.slug}")
        print(r.stdout.strip(), r.stderr.strip())
    else:
        r = kaggle("kernels", "output", f"{args.user}/{args.slug}", "-p", args.out)
        print(r.stdout.strip()[-2000:], r.stderr.strip()[-2000:])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
