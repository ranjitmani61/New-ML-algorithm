"""Fine-tune on short three-person problems (answer: the largest final amount) under one condition;
evaluate on longer ones. This is the test where answers cannot be composed (see multi.py).

    python3 run_multi.py --model MODEL_DIR --condition self_improve --seed 9 --out OUT_DIR

Conditions
  plain          short labeled problems only (1-4 events)
  cancel_spread  the method: half of the short problems get cancelling pairs (up to 16 events)
  oracle         true labels for 1-16 events (needs a solver; the ceiling)
  self_improve   direct self-improvement (Lee et al. 2025, arXiv 2502.01612), adapted:
                 * --si-models models are trained on the same short labeled problems with different
                   seeds (model 0 is trained exactly like plain);
                 * each round draws --si-n unlabeled problems at the next lengths (--si-step more
                   events per round, up to --expand-max), labels them with every model (greedy),
                   and keeps a label only if all models give the same number (majority voting with
                   full agreement, as in Lee et al.'s maze runs);
                 * every model then trains --si-steps more steps on all data so far, sampling the
                   newest round's data with probability 0.5 and the rest uniformly;
                 * the true labels of the unlabeled problems are logged for diagnosis only and are
                   never used for training or filtering.
                 --si-max-rounds 0 trains and evaluates the models only (the pilot), and
                 --si-probe-ks measures the vote on fresh problems at those lengths.

The model answers with the number only; the loss is on the answer tokens only. Prompt format,
batching, generation and the strict parse are run_llm.py's (imported, unchanged).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import time

import numpy as np
import torch

import multi as M
import run_llm as R

CONDITIONS = ("plain", "cancel_spread", "oracle", "self_improve")
HERE = os.path.dirname(os.path.abspath(__file__))


class Labeled:
    """A training example with an explicit label (true or self-generated) for R.encode_example."""

    def __init__(self, text, label):
        self._text, self._label = text, label

    def text(self):
        return self._text

    def answer(self):
        return self._label


def build_train(rng, condition, n, label_max, expand_max):
    out = []
    for _ in range(n):
        kmax = expand_max if condition == "oracle" else label_max
        p = M.make_multi(rng, rng.randint(1, kmax))
        if condition == "cancel_spread" and rng.random() < 0.5:
            q = M.cancel_spread(rng, p, expand_max)
            assert q.answer() == p.answer() and q.valid()  # a check only: the construction never simulates
            p = q
        out.append(p)
    return out


def build_eval(seed, ks, per_k, heldout=False):
    sets = {}
    for k in ks:
        rng = random.Random(900_000 + 1000 * seed + k + (500 if heldout else 0))
        sets[k] = [M.make_multi(rng, k, heldout=heldout) for _ in range(per_k)]
    return sets


def si_schedule(label_max, expand_max, step):
    rounds, lo = [], label_max + 1
    while lo <= expand_max:
        hi = min(lo + step - 1, expand_max)
        rounds.append((lo, hi))
        lo = hi + 1
    return rounds


def model_seed(seed, m):
    return seed if m == 0 else 10_000 * m + seed


def vote(outputs_by_model):
    """Full agreement: the label every model gives, or None (also None if any output fails to parse)."""
    out = []
    for preds in zip(*[[R.parse(o) for o in outs] for outs in outputs_by_model]):
        out.append(preds[0] if preds[0] is not None and all(p == preds[0] for p in preds) else None)
    return out


def si_sampler(n_old, n_new, batch, seed):
    """Batch indices: each slot comes from the newest round's data (indices n_old..n_old+n_new-1)
    with probability 0.5, otherwise uniformly from all older data (0..n_old-1)."""
    g = np.random.default_rng(seed)

    def sample(step):
        new = g.random(batch) < 0.5 if n_new else np.zeros(batch, bool)
        return [int(n_old + g.integers(n_new)) if b else int(g.integers(n_old)) for b in new]
    return sample


def stats_of(enc, examples):
    plens = np.array([e[2] for e in enc])
    return {"n": len(examples), "prompt_tokens_mean": float(plens.mean()),
            "prompt_tokens_p50_p90_max": [int(np.percentile(plens, 50)), int(np.percentile(plens, 90)), int(plens.max())]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--condition", choices=CONDITIONS, required=True)
    ap.add_argument("--family", default="multi", choices=("multi",))  # for the kernel runner's folder names
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=24000)
    ap.add_argument("--label-max", type=int, default=4)
    ap.add_argument("--expand-max", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--eval-ks", nargs="+", type=int, default=[4, 8, 12, 16])
    ap.add_argument("--eval-per-k", type=int, default=500)
    ap.add_argument("--si-models", type=int, default=3)
    ap.add_argument("--si-step", type=int, default=2)
    ap.add_argument("--si-n", type=int, default=4000)
    ap.add_argument("--si-steps", type=int, default=500)
    ap.add_argument("--si-max-rounds", type=int, default=-1, help="-1: the whole schedule; 0: no rounds (pilot)")
    ap.add_argument("--si-probe-ks", nargs="*", type=int, default=[])
    ap.add_argument("--si-probe-n", type=int, default=500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    t0 = time.time()

    def log(obj):
        print(json.dumps(obj), flush=True)

    def fresh_model(mseed):
        torch.manual_seed(mseed)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
        model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                                 "gate_proj", "up_proj", "down_proj"],
                                                 task_type="CAUSAL_LM"))
        for prm in model.parameters():  # LoRA weights in fp32 for stable fp16 training
            if prm.requires_grad:
                prm.data = prm.data.float()
        return model

    def train(model, enc, steps, warmup, sampler, tag):
        """run_llm.py's training step and schedule; warmup=0 gives a plain linear decay."""
        params = [q for q in model.parameters() if q.requires_grad]
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (min(1.0, (s + 1) / warmup) if warmup else 1.0) * max(0.0, 1 - s / steps))
        scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
        model.train()
        for step in range(steps):
            ids, lab, att = (t.to(device) for t in R.collate([enc[i][:2] for i in sampler(step)], tok.pad_token_id))
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            if (step + 1) % max(steps // 10, 1) == 0:
                log({"train": tag, "step": step + 1, "loss": round(loss.item(), 4), "elapsed": round(time.time() - t0)})
        model.eval()

    def order_sampler(n, seed):
        order = np.random.default_rng(seed).permutation(n)
        return lambda step: [order[(step * args.batch + j) % n] for j in range(args.batch)]

    def evaluate(model, sets_all, gen_file=None):
        metrics = {}
        for (tag, k), probs in sets_all.items():
            outs = R.generate(model, tok, probs, device)
            preds = [R.parse(o) for o in outs]
            ok = [pr == p.answer() for pr, p in zip(preds, probs)]
            lenient = [R.parse_lenient(o) == p.answer() for o, p in zip(outs, probs)]
            metrics[f"{tag}/{k}"] = {"accuracy": float(np.mean(ok)), "accuracy_lenient": float(np.mean(lenient)),
                                     "parse_failures": int(sum(x is None for x in preds)), "n": len(probs)}
            if gen_file is not None:
                for p, o, pr, c in zip(probs, outs, preds, ok):
                    gen_file.write(json.dumps({"set": f"{tag}/{k}", "text": p.text(), "answer": p.answer(),
                                               "output": o, "parsed": pr, "correct": c}) + "\n")
        return metrics

    # ---------------------------------------------------------------- data
    rng = random.Random(args.seed)
    train_set = build_train(rng, args.condition, args.n_train, args.label_max, args.expand_max)
    examples = [Labeled(p.text(), p.answer()) for p in train_set]
    enc = [R.encode_example(tok, e, args.max_len) for e in examples]
    train_keys = {M.key(p) for p in train_set}
    stats = {**stats_of(enc, examples), "changes_mean": float(np.mean([p.n_changes() for p in train_set])),
             "answer_max": max(p.answer() for p in train_set), "truncated": int(sum(len(e[0]) >= args.max_len for e in enc))}
    log({"train_stats": stats})
    sets_all = {}
    for tag, heldout, ks in (("train_vocab", False, args.eval_ks), ("heldout_vocab", True, [12])):
        for k, probs in build_eval(args.seed, ks, args.eval_per_k, heldout).items():
            sets_all[(tag, k)] = probs

    # ---------------------------------------------------------------- training
    si = {}
    n_models = args.si_models if args.condition == "self_improve" else 1
    models = []
    for m in range(n_models):
        model = fresh_model(model_seed(args.seed, m))
        train(model, enc, args.steps, 50, order_sampler(len(enc), model_seed(args.seed, m)), f"m{m}/r0")
        models.append(model)
    if args.condition == "self_improve":
        si = {"schedule": si_schedule(args.label_max, args.expand_max, args.si_step), "rounds": [], "probe": {},
              "round0": {}}
        rows_file = gzip.open(os.path.join(args.out, "si_data.jsonl.gz"), "wt")
        for m, model in enumerate(models):  # round-0 models: trained like plain (model 0 exactly)
            si["round0"][m] = evaluate(model, sets_all)
            log({"round0": m, "metrics": si["round0"][m]})
        for k in args.si_probe_ks:  # the vote on fresh problems, without training on them
            probe_rng = random.Random(800_000 + 1000 * args.seed + k)
            probs = [M.make_multi(probe_rng, k) for _ in range(args.si_probe_n)]
            outs = [R.generate(model, tok, probs, device) for model in models]
            voted = vote(outs)
            acc = [float(np.mean([R.parse(o) == p.answer() for o, p in zip(out, probs)])) for out in outs]
            kept = [(v, p.answer()) for v, p in zip(voted, probs) if v is not None]
            si["probe"][k] = {"n": len(probs), "per_model_accuracy": acc, "kept": len(kept),
                              "kept_label_accuracy": float(np.mean([v == a for v, a in kept])) if kept else None}
            log({"probe": k, **si["probe"][k]})
        rounds = si["schedule"] if args.si_max_rounds < 0 else si["schedule"][:args.si_max_rounds]
        data_enc, data_examples = list(enc), list(examples)  # D0, then each round's kept examples
        for r, (lo, hi) in enumerate(rounds, start=1):
            pool_rng = random.Random(700_000 + 1000 * args.seed + 10 * r)
            pool = [M.make_multi(pool_rng, pool_rng.randint(lo, hi)) for _ in range(args.si_n)]
            outs = [R.generate(model, tok, pool, device) for model in models]
            voted = vote(outs)
            new = [Labeled(p.text(), v) for p, v in zip(pool, voted) if v is not None]  # self-generated labels only
            for i, p in enumerate(pool):
                rows_file.write(json.dumps({"round": r, "text": p.text(), "events": p.n_changes(),
                                            "outputs": [o[i] for o in outs], "voted": voted[i],
                                            "true_answer_for_diagnosis_only": p.answer()}) + "\n")
            train_keys |= {M.key(p) for p, v in zip(pool, voted) if v is not None}
            new_enc = [R.encode_example(tok, e, args.max_len) for e in new]
            kept = [(v, p.answer()) for p, v in zip(pool, voted) if v is not None]
            info = {"round": r, "events": [lo, hi], "pool": len(pool), "kept": len(new),
                    "kept_label_accuracy_diagnostic": float(np.mean([v == a for v, a in kept])) if kept else None,
                    "per_model_pool_accuracy_diagnostic": [float(np.mean([R.parse(o) == p.answer() for o, p in zip(out, pool)]))
                                                           for out in outs],
                    "truncated": int(sum(len(e[0]) >= args.max_len for e in new_enc))}
            n_old = len(data_enc)
            data_enc += new_enc
            data_examples += new
            for m, model in enumerate(models):
                train(model, data_enc, args.si_steps, 0,
                      si_sampler(n_old, len(new_enc), args.batch, 1_000 * model_seed(args.seed, m) + r), f"m{m}/r{r}")
            info["elapsed"] = round(time.time() - t0)
            si["rounds"].append(info)
            log({"si_round": info})
        rows_file.close()
        stats["si_examples_total"] = len(data_examples)
    for m, model in enumerate(models):
        model.save_pretrained(os.path.join(args.out, "adapter" if m == 0 else f"adapter_m{m}"))
    train_s = time.time() - t0

    # ---------------------------------------------------------------- evaluation (model 0 is the result)
    leak = {f"{tag}/{k}": sum(M.key(p) in train_keys for p in probs) for (tag, k), probs in sets_all.items()}
    with open(os.path.join(args.out, "generations.jsonl"), "w") as g:
        metrics = evaluate(models[0], sets_all, g)
    log({"eval_model0": metrics})
    if args.condition == "self_improve":
        si["final_other_models"] = {}
        for m, model in enumerate(models[1:], start=1):
            si["final_other_models"][m] = evaluate(model, sets_all)
            log({"eval_model": m, "metrics": si["final_other_models"][m]})
    files = ("multi.py", "run_multi.py", "run_llm.py", "problems.py")
    res = {"args": vars(args), "device": device, "torch": torch.__version__, "train_stats": stats,
           "leakage_exact_overlap": leak, "metrics": metrics, "self_improve": si, "train_seconds": round(train_s),
           "total_seconds": round(time.time() - t0),
           "code_sha256": {n: hashlib.sha256(open(os.path.join(HERE, n), "rb").read()).hexdigest() for n in files}}
    json.dump(res, open(os.path.join(args.out, "metrics.json"), "w"), indent=1)
    log({"done": args.condition, "seed": args.seed, "metrics": metrics})


if __name__ == "__main__":
    main()
