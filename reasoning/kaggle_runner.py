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
DATASETS = ["thedevastator/grade-school-math-8k-q-a", "open-benchmarks/math-500-measuring-mathematical-problem-solving",
            "angelirodriguez/hendrycks-math-dataset"]
INPUT_ROOT = os.environ.get("KAGGLE_INPUT_ROOT", "/kaggle/input")
SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."

# Question sets. Selection is deterministic and shared with benchmarks.py.
SETS = {
    "gsm8k_dev": {"file": "main_train.csv", "pick": ("sample", 400, 20261101), "max_new": 640},
    "gsm8k_test": {"file": "main_test.csv", "pick": ("all",), "max_new": 640},
    "math_dev": {"file": "math_500_test.csv", "pick": ("sample", 150, 20261101), "max_new": 1280},
    "math_test": {"file": "math_500_test.csv", "pick": ("complement", 150, 20261101), "max_new": 1280},
    "smoke": {"file": "math_500_test.csv", "pick": ("sample", 12, 7), "max_new": 768},
    # Hard setting: MATH level 5. Development from the train split; the sealed test is every
    # test-split level-5 problem that is not in MATH-500 (MATH-500 was used by the sets above).
    "hard_dev": {"math_json": "train", "pick": ("sample", 500, 20261101), "max_new": 2048},
    "hard_test": {"math_json": "test", "pick": ("all",), "max_new": 2048},
    "hard_smoke": {"math_json": "train", "pick": ("sample", 12, 7), "max_new": 2048},
}


def _norm_text(s):
    return " ".join(s.split())


def read_math_json(split, level="Level 5"):
    """Level-5 problems of one split of the original MATH release, minus MATH-500, sorted by path."""
    base = sorted(glob.glob(f"{INPUT_ROOT}/**/MATH/{split}", recursive=True))
    if not base:
        raise FileNotFoundError(f"MATH/{split} under {INPUT_ROOT}")
    m500_rows, _ = read_rows(find_file("math_500_test.csv"))
    m500 = {_norm_text(r["question"]) for r in m500_rows}
    rows = []
    for path in sorted(glob.glob(f"{base[0]}/*/*.json")):
        d = json.load(open(path, encoding="utf-8"))
        if d.get("level") != level or _norm_text(d["problem"]) in m500:
            continue
        rel = f"{split}/{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}"
        rows.append({"index": rel, "question": d["problem"], "solution": d["solution"], "type": d.get("type")})
    return rows


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
    if "math_json" in SETS[name]:
        return [{"id": f"math5:{rows[i]['index']}", "set": name, **rows[i]} for i in idx]
    prefix = name.split("_")[0]
    source = SETS[name]["file"].rsplit(".", 1)[0]
    return [{"id": f"{prefix}:{source}:{rows[i]['index']}", "set": name, **rows[i]} for i in idx]


def load_set(name):
    spec = SETS[name]
    rows = read_math_json(spec["math_json"]) if "math_json" in spec else read_rows(find_file(spec["file"]))[0]
    return select(name, rows)


# ---------------------------------------------------------------- Kaggle side


def find_file(name):
    hits = glob.glob(f"{INPUT_ROOT}/**/{name}", recursive=True)
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


def worker_vllm(gpu, shard_path, out_prefix, cfg):
    """Same outputs as worker(), generated with vLLM (continuous batching, no padding waste).

    Signals come from vLLM's top-20 logprobs. With logprobs_mode="raw_logprobs" they are taken on
    the model's raw distribution, before temperature and top-p, as in the HF worker.
    """
    import numpy as np
    import vllm
    from vllm import LLM, SamplingParams

    items = [json.loads(line) for line in open(shard_path)]
    sys.path.insert(0, "/kaggle/working")
    import benchmarks as bm

    model_dir = cfg.get("model_dir") or find_model_dir()
    kwargs = dict(model=model_dir, dtype="float16", max_model_len=4096, gpu_memory_utilization=0.90,
                  seed=cfg["seed"] * 1009 + gpu, max_logprobs=20, max_num_seqs=cfg.get("max_num_seqs", 256))
    mode = "raw_logprobs"
    try:
        llm = LLM(logprobs_mode=mode, **kwargs)
    except TypeError:  # older vLLM without logprobs_mode: record that signals are processed
        mode = "default(no logprobs_mode)"
        llm = LLM(**kwargs)
    tok = llm.get_tokenizer()
    n, k = cfg["samples"], 20
    features = open(f"{out_prefix}.features.jsonl", "w")
    out_json = gzip.open(f"{out_prefix}.jsonl.gz", "wt")
    conf_all, lp_all, offsets = [], [], [0]
    start, generated, done = time.time(), 0, 0
    chunk = cfg.get("chunk_questions", 48)
    for c0 in range(0, len(items), chunk):
        if time.time() - cfg["started"] > cfg["time_budget_s"]:
            print(json.dumps({"gpu": gpu, "stopped": "time budget", "done": done}), flush=True)
            break
        batch = items[c0:c0 + chunk]
        prompts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": b["question"]}],
                                           tokenize=False, add_generation_prompt=True) for b in batch]
        params = [SamplingParams(n=n, temperature=cfg["temperature"], top_p=cfg["top_p"],
                                 max_tokens=SETS[b["set"]]["max_new"], logprobs=k, detokenize=False) for b in batch]
        outs = llm.generate(prompts, params, use_tqdm=False)
        for item, res in zip(batch, outs):
            samples, summaries = [], []
            for o in res.outputs:
                ids = list(o.token_ids)
                length = len(ids)
                conf = np.empty(length, np.float32)
                lps = np.empty(length, np.float32)
                for t, (tid, entry) in enumerate(zip(ids, o.logprobs)):
                    vals = sorted((v.logprob for v in entry.values()), reverse=True)[:k]
                    conf[t] = -float(np.mean(vals))
                    lps[t] = entry[tid].logprob
                finished = o.finish_reason == "stop"
                text = tok.decode(ids, skip_special_tokens=True)
                conf_all.append(conf.astype("float16"))
                lp_all.append(lps.astype("float16"))
                offsets.append(offsets[-1] + length)
                generated += length
                summaries.append(bm.summarize_attempt(text, length, finished, conf, lps))
                samples.append({"text": text, "tokens": int(length), "finished": finished,
                                "sum_logprob": float(lps.sum()), "mean_conf": float(conf.mean()) if length else 0.0,
                                "array": len(offsets) - 2})
            out_json.write(json.dumps({**{k2: item[k2] for k2 in ("id", "set", "question", "solution")},
                                       "samples": samples, "prompt_tokens": len(res.prompt_token_ids)}) + "\n")
            features.write(json.dumps({"id": item["id"], "set": item["set"], "gold": bm.gold_answer(item),
                                       "samples": summaries}) + "\n")
            done += 1
        out_json.flush()
        features.flush()
        elapsed = time.time() - start
        print(json.dumps({"gpu": gpu, "done": done, "of": len(items), "tokens": generated,
                          "tok_per_s": round(generated / elapsed, 1)}), flush=True)
    out_json.close()
    features.close()
    np.savez_compressed(f"{out_prefix}.npz", conf=np.concatenate(conf_all) if conf_all else np.zeros(0, "float16"),
                        logprob=np.concatenate(lp_all) if lp_all else np.zeros(0, "float16"),
                        offsets=np.array(offsets, dtype=np.int64))
    import torch

    meta = {"gpu": gpu, "items": len(items), "completed": done, "tokens": generated, "engine": "vllm",
            "vllm": vllm.__version__, "logprobs_mode": mode, "seconds": round(time.time() - start, 1),
            "torch": torch.__version__, "device": torch.cuda.get_device_name(0)}
    with open(f"{out_prefix}.meta.json", "w") as fh:
        json.dump(meta, fh)


def install_vllm(spec):
    """pip install vLLM inside the kernel (needs internet); returns the installed version."""
    t = time.time()
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", spec], capture_output=True, text=True)
    with open("/kaggle/working/pip_vllm.log", "w") as fh:
        fh.write(r.stdout + "\n" + r.stderr)
    v = subprocess.run([sys.executable, "-c", "import vllm, torch; print(vllm.__version__, torch.__version__)"],
                       capture_output=True, text=True)
    info = {"pip_exit": r.returncode, "seconds": round(time.time() - t), "versions": v.stdout.strip(),
            "import_error": v.stderr.strip()[-800:]}
    print(json.dumps({"install": info}), flush=True)
    if r.returncode != 0 or v.returncode != 0:
        raise RuntimeError(f"vLLM install failed: {info} | {r.stderr[-800:]}")
    return info


def kernel_main(cfg):
    import torch

    n_gpus = torch.cuda.device_count()  # read before a vLLM install may replace torch on disk
    cfg["started"] = time.time()
    os.makedirs("/kaggle/working/pools", exist_ok=True)
    import base64

    with open("/kaggle/working/benchmarks.py", "wb") as fh:
        fh.write(base64.b64decode(BENCHMARKS_B64))
    cfg["model_dir"] = find_model_dir()
    if cfg.get("engine") == "vllm":
        cfg["install"] = install_vllm(cfg.get("vllm_spec", "vllm"))
    print(json.dumps({"model_dir": cfg["model_dir"], "inputs": sorted(os.listdir("/kaggle/input")),
                      "gpus": n_gpus}), flush=True)
    items = []
    for name in cfg["sets"]:
        items.extend(load_set(name))
    gpus = max(1, n_gpus)
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


def push(slug, sets, samples, rows, temperature, top_p, seed, budget_hours, user, engine="hf", vllm_spec="vllm"):
    import tempfile

    cfg = {"sets": sets, "samples": samples, "rows": rows, "temperature": temperature, "top_p": top_p,
           "seed": seed, "time_budget_s": int(budget_hours * 3600), "engine": engine}
    if engine == "vllm":
        cfg["vllm_spec"] = vllm_spec
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
            "enable_internet": engine == "vllm", "dataset_sources": DATASETS, "kernel_sources": [],
            "competition_sources": [], "model_sources": [MODEL_SOURCE], "machine_shape": "NvidiaTeslaT4"}
    with open(os.path.join(d, "kernel-metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    r = kaggle("kernels", "push", "-p", d)
    print(r.stdout, r.stderr)
    return cfg


def main(argv):
    if len(argv) >= 2 and argv[1] == "--worker":
        cfg = json.loads(argv[5])
        (worker_vllm if cfg.get("engine") == "vllm" else worker)(int(argv[2]), argv[3], argv[4], cfg)
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
    p.add_argument("--engine", choices=["hf", "vllm"], default="hf")
    p.add_argument("--vllm-spec", default="vllm")
    for name in ("status", "fetch", "log", "fetch-log"):
        q = sub.add_parser(name)
        q.add_argument("--slug", required=True)
        q.add_argument("--user", default=os.environ.get("KAGGLE_USERNAME", "epicnexus"))
        if name in ("fetch", "fetch-log"):
            q.add_argument("--out", required=True)
    args = ap.parse_args(argv[1:])
    if args.cmd == "push":
        print(json.dumps(push(args.slug, args.sets, args.samples, args.rows, args.temperature, args.top_p,
                              args.seed, args.budget_hours, args.user, args.engine, args.vllm_spec)))
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
