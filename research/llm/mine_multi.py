"""Certified rewrite mining (tournament ledger #33) on the three-person language-model test.

    python3 mine_multi.py --model MODEL_DIR --seed 9 --m0-root /kaggle/input --out OUT_DIR

M0 is the final's plain model for this seed (`*_plain_multi_s<seed>/adapter`, trained on 1-4
events; results/multi_final). Nothing else from the final is used.

1. Edits never read an answer. A segment is an event (person 0-2, gain or loss, 1-9): 54 segments.
   Edits: insert one segment; insert two at independent random positions ("spread"); insert two as
   an adjacent block. 54 + 2 x 54^2 = 5,886 edits. An inserted event is rendered with multi.py's
   templates and the problem's own people and item.
2. Certification, with M0's greedy answers only: apply an edit to problems with 1-2 events (so the
   edited problems have 2-4 events, inside M0's training lengths) and count how often M0's answer
   equals the original problem's true label. Stage 1: --cert-n1 problems; an edit with more than
   --cert-m1 mismatches is rejected. Stage 2: up to --cert-n problems in total; certified if the
   score is at least M0's accuracy on unedited problems of the same lengths minus --cert-margin.
   Gate: M0 must get at least --gate on unedited problems with 3-4 events, or the run stops.
3. Training: run_multi.py's plain recipe, except that half of the short problems are lengthened
   with random certified edits up to 16 events and keep their short label (never recomputed).

The true answers of edited problems are computed only to measure the certificates and the label
error of the training data afterwards; they never decide anything.
"""

from __future__ import annotations

import argparse
import glob
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
import run_multi as RM
from problems import NAMES_TRAIN

HERE = os.path.dirname(os.path.abspath(__file__))
SEGMENTS = [(who, kind, n) for who in range(M.N_PEOPLE) for kind in ("gain", "loss") for n in range(1, 10)]
EDITS = ([("ins1", (s,)) for s in SEGMENTS]
         + [(k, (s, t)) for k in ("spread", "block") for s in SEGMENTS for t in SEGMENTS])


def size(edit):
    return len(edit[1])


def exact(edit):
    """Evaluation only: does the edit keep the largest final amount of every problem?"""
    if edit[0] == "ins1":
        return False
    (w1, k1, n1), (w2, k2, n2) = edit[1]
    return w1 == w2 and k1 != k2 and n1 == n2


def apply_edit(rng, p, edit):
    q = M._copy(p)
    evs = [M.Event(who, kind, n, M._render(rng, kind, n, q.people, who, q.item, NAMES_TRAIN))
           for who, kind, n in edit[1]]
    if edit[0] == "block":
        i = rng.randint(0, len(q.events))
        q.events[i:i] = evs
    else:
        for e in evs:
            q.events.insert(rng.randint(0, len(q.events)), e)
    return q


def lengthen(rng, p, certified, cap):
    """Random certified edits until a random length budget (0 .. cap - len) is used up."""
    budget, q = rng.randint(0, cap - p.n_changes()), p
    while True:
        fits = [e for e in certified if size(e) <= budget]
        if not fits:
            return q
        e = fits[rng.randrange(len(fits))]
        q = apply_edit(rng, q, e)
        budget -= size(e)


def reference(answers, rc, ref_n):
    """Accuracy of `answers` on unedited problems of 1-2 events plus k, for edit sizes k = 1, 2."""
    ref = {}
    for k in (1, 2):
        probs = [M.make_multi(rc, rc.randint(1, 2) + k) for _ in range(ref_n)]
        ref[k] = float(np.mean([a == p.answer() for a, p in zip(answers(probs), probs)]))
    return ref


def certify(answers, rc, edits, ref, n1, m1, n, margin):
    """Score edits with `answers` (problems -> parsed answers). Returns (table, certified, n_stage2)."""
    rows = {e: {"miss": 0, "n": 0, "inv": 0, "valid": 0} for e in edits}

    def run_stage(es, n_each):
        items = []
        for e in es:
            for _ in range(n_each):
                p = M.make_multi(rc, rc.randint(1, 2))
                items.append((e, p.answer(), apply_edit(rc, p, e)))
        preds = answers([q for _, _, q in items])
        for (e, y, q), a in zip(items, preds):
            r = rows[e]
            r["n"] += 1
            r["miss"] += a != y
            r["inv"] += q.answer() == y      # evaluation only
            r["valid"] += q.valid()          # evaluation only

    run_stage(edits, n1)
    stage2 = [e for e in edits if rows[e]["miss"] <= m1]
    run_stage(stage2, n - n1)
    s2 = set(stage2)
    table, certified = [], []
    for e in edits:
        r = rows[e]
        score = 1 - r["miss"] / r["n"]
        ok = e in s2 and score >= ref[size(e)] - margin
        if ok:
            certified.append(e)
        table.append({"edit": [e[0], [list(x) for x in e[1]]], "n": r["n"], "score": score, "certified": ok,
                      "exact": exact(e), "inv": r["inv"] / r["n"], "valid": r["valid"] / r["n"]})
    return table, certified, len(stage2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--condition", default="mined", choices=("mined",))
    ap.add_argument("--family", default="multi", choices=("multi",))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--m0-root", default="/kaggle/input")
    ap.add_argument("--m0-adapter", default="")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=24000)
    ap.add_argument("--label-max", type=int, default=4)
    ap.add_argument("--expand-max", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--eval-ks", nargs="+", type=int, default=[4, 8, 12, 16])
    ap.add_argument("--eval-per-k", type=int, default=500)
    ap.add_argument("--cert-n1", type=int, default=8)
    ap.add_argument("--cert-m1", type=int, default=1)
    ap.add_argument("--cert-n", type=int, default=64)
    ap.add_argument("--cert-margin", type=float, default=0.05)
    ap.add_argument("--cert-ref-n", type=int, default=1000)
    ap.add_argument("--cert-batch", type=int, default=128)
    ap.add_argument("--cert-max-edits", type=int, default=0, help="smoke tests only: a random subset of edits")
    ap.add_argument("--gate", type=float, default=0.90)
    ap.add_argument("--cert-predictor", default="model", choices=("model", "oracle"),
                    help="oracle = true answers, for local smoke tests only (never in a real run)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    t0 = time.time()
    res = {"args": vars(args), "device": device, "torch": torch.__version__}
    files = ("multi.py", "run_multi.py", "run_llm.py", "problems.py", "mine_multi.py")
    res["code_sha256"] = {n: hashlib.sha256(open(os.path.join(HERE, n), "rb").read()).hexdigest() for n in files}

    def log(obj):
        print(json.dumps(obj), flush=True)

    def save():
        json.dump(res, open(os.path.join(args.out, "metrics.json"), "w"), indent=1)

    # ------------------------------------------------ M0: the final's plain adapter for this seed
    m0 = args.m0_adapter
    if not m0:
        hits = glob.glob(os.path.join(args.m0_root, "**", f"*_plain_multi_s{args.seed}", "adapter", "adapter_config.json"),
                         recursive=True)
        if len(hits) != 1:
            res["stopped"] = f"M0 adapter not found exactly once: {hits}"
            log(res); save(); return
        m0 = os.path.dirname(hits[0])
    res["m0_adapter"] = m0
    res["m0_adapter_sha256"] = hashlib.sha256(open(os.path.join(m0, "adapter_model.safetensors"), "rb").read()).hexdigest()
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model0 = PeftModel.from_pretrained(base, m0).to(device)
    model0.eval()

    def answers(problems):
        return [R.parse(o) for o in R.generate(model0, tok, problems, device, batch=args.cert_batch)]

    # ------------------------------------------------ certification (M0's answers vs short labels)
    if args.cert_predictor == "oracle":  # local smoke tests only: a perfect M0
        answers = lambda probs: [p.answer() for p in probs]  # noqa: E731
    edits = list(EDITS)
    if args.cert_max_edits:
        edits = [edits[i] for i in sorted(random.Random(args.seed).sample(range(len(edits)), args.cert_max_edits))]
    rc = random.Random(700_000 + args.seed)
    ref = res["ref"] = reference(answers, rc, args.cert_ref_n)
    log({"ref": ref, "elapsed": round(time.time() - t0)})
    if ref[2] < args.gate:
        res["stopped"] = f"gate failed: M0 {ref[2]:.3f} on unedited 3-4-event problems (needs {args.gate})"
        log(res); save(); return
    table, certified, n_stage2 = certify(answers, rc, edits, ref, args.cert_n1, args.cert_m1, args.cert_n,
                                         args.cert_margin)
    n_exact = sum(exact(e) for e in edits)
    both = sum(exact(e) for e in certified)
    res["certification"] = {
        "edits": len(edits), "stage2": n_stage2, "certified": len(certified), "exact": n_exact,
        "certified_and_exact": both, "precision": both / len(certified) if certified else None,
        "recall": both / n_exact if n_exact else None,
        "certified_not_exact": [[t["edit"], round(t["score"], 3), round(t["inv"], 3)] for t in table
                                if t["certified"] and not t["exact"]],
        "generations": sum(t["n"] for t in table) + 2 * args.cert_ref_n, "seconds": round(time.time() - t0)}
    with gzip.open(os.path.join(args.out, "certification.jsonl.gz"), "wt") as f:
        for t in table:
            f.write(json.dumps(t) + "\n")
    log({"certification": res["certification"]})
    save()
    del model0, base
    if device == "cuda":
        torch.cuda.empty_cache()
    if not certified:
        res["stopped"] = "no edit certified"
        log(res); save(); return

    # ------------------------------------------------ training data: lengthened, labels inherited
    rng = random.Random(args.seed)
    train_set, examples, lab_ok = [], [], []
    for _ in range(args.n_train):
        p = M.make_multi(rng, rng.randint(1, args.label_max))
        y = p.answer()
        if rng.random() < 0.5:
            p = lengthen(rng, p, certified, args.expand_max)
        train_set.append(p)
        examples.append(RM.Labeled(p.text(), y))
        lab_ok.append(p.answer() == y)   # evaluation only
    enc = [R.encode_example(tok, e, args.max_len) for e in examples]
    train_keys = {M.key(p) for p in train_set}
    stats = {**RM.stats_of(enc, examples), "changes_mean": float(np.mean([p.n_changes() for p in train_set])),
             "answer_max": max(e.answer() for e in examples),
             "truncated": int(sum(len(e[0]) >= args.max_len for e in enc)),
             "label_correct_rate": float(np.mean(lab_ok)), "valid_rate": float(np.mean([p.valid() for p in train_set]))}
    res["train_stats"] = stats
    log({"train_stats": stats})
    sets_all = {}
    for tag, heldout, ks in (("train_vocab", False, args.eval_ks), ("heldout_vocab", True, [12])):
        for k, probs in RM.build_eval(args.seed, ks, args.eval_per_k, heldout).items():
            sets_all[(tag, k)] = probs

    # ------------------------------------------------ run_multi.py's training and evaluation, unchanged
    def fresh_model(mseed):
        torch.manual_seed(mseed)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
        model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                                 "gate_proj", "up_proj", "down_proj"],
                                                 task_type="CAUSAL_LM"))
        for prm in model.parameters():
            if prm.requires_grad:
                prm.data = prm.data.float()
        return model

    def train(model, enc, steps, warmup, sampler, tag):
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

    model = fresh_model(RM.model_seed(args.seed, 0))
    train(model, enc, args.steps, 50, order_sampler(len(enc), RM.model_seed(args.seed, 0)), "m0/r0")
    model.save_pretrained(os.path.join(args.out, "adapter"))
    res["train_seconds"] = round(time.time() - t0)
    res["leakage_exact_overlap"] = {f"{tag}/{k}": sum(M.key(p) in train_keys for p in probs)
                                    for (tag, k), probs in sets_all.items()}
    with open(os.path.join(args.out, "generations.jsonl"), "w") as g:
        res["metrics"] = evaluate(model, sets_all, g)
    res["total_seconds"] = round(time.time() - t0)
    log({"eval_model0": res["metrics"]})
    save()


if __name__ == "__main__":
    main()
