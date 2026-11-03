"""Fine-tune a small instruct model on short word problems under one condition; evaluate on longer.

    python3 run_llm.py --model MODEL_DIR --condition cancel_spread --seed 0 --out OUT_DIR

Conditions: plain, irrelevant, inactive, cancel_spread, cancel_adjacent, oracle (true labels for
all lengths). --steps 0 evaluates the base model unchanged (headroom check).

The model answers with the number only. Loss is computed on the answer tokens only. Everything
needed to recompute the metrics is saved: raw generations (generations.jsonl), the trained
adapter, training-set statistics, the leakage check and metrics.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time

import numpy as np
import torch

import problems as P

SYSTEM = "Solve the word problem. Answer with the final number only."
CONDITIONS = ("plain", "irrelevant", "inactive", "cancel_spread", "cancel_adjacent", "oracle")
ANSWER = re.compile(r"^\s*(-?\d+)")
ANY_INT = re.compile(r"-?\d+")


def build_train(rng, condition, family, n, label_max, expand_max):
    out = []
    for _ in range(n):
        kmax = expand_max if condition == "oracle" else label_max
        p = P.make_problem(rng, rng.randint(1, kmax), family)
        if condition in P.EXPANSIONS and rng.random() < 0.5:
            q = P.EXPANSIONS[condition](rng, p, expand_max, family)  # raises for cancel_spread + mixed
            assert q.answer() == p.answer() and q.valid()
            p = q
        out.append(p)
    return out


def build_eval(seed, family, ks, per_k, heldout=False):
    sets = {}
    for k in ks:
        rng = random.Random(900_000 + 1000 * seed + k + (500 if heldout else 0))
        sets[k] = [P.make_problem(rng, k, family, heldout=heldout) for _ in range(per_k)]
    return sets


def chat_prompt(tok, text):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def encode_example(tok, p, max_len):
    prompt = tok(chat_prompt(tok, p.text()), add_special_tokens=False)["input_ids"]
    target = tok(str(p.answer()), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids = (prompt + target)[-max_len:]
    labels = ([-100] * len(prompt) + target)[-max_len:]
    return ids, labels, len(prompt)


def collate(batch, pad_id):
    L = max(len(i) for i, _ in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for r, (i, l) in enumerate(batch):  # right padding for training
        ids[r, :len(i)] = torch.tensor(i)
        lab[r, :len(l)] = torch.tensor(l)
        att[r, :len(i)] = 1
    return ids, lab, att


@torch.no_grad()
def generate(model, tok, problems, device, batch=64, max_new=6):
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(problems), batch):
        chunk = problems[i:i + batch]
        enc = tok([chat_prompt(tok, p.text()) for p in chunk], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        g = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], max_new_tokens=max_new,
                           do_sample=False, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs


def parse(text):
    """Strict: the output must start with the number (the trained format)."""
    m = ANSWER.match(text)
    return int(m.group(1)) if m else None


def parse_lenient(text):
    """Lenient: the first integer anywhere (reported for the base model, which may add words)."""
    m = ANY_INT.search(text)
    return int(m.group(0)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--condition", choices=CONDITIONS, required=True)
    ap.add_argument("--family", choices=("additive", "mixed"), default="additive")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=24000)
    ap.add_argument("--label-max", type=int, default=4)
    ap.add_argument("--expand-max", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--eval-ks", nargs="+", type=int, default=[4, 8, 12, 16])
    ap.add_argument("--eval-per-k", type=int, default=500)
    ap.add_argument("--max-new", type=int, default=0, help="new tokens at evaluation; default 6, or 16 for --steps 0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    t0 = time.time()
    rng = random.Random(args.seed)
    train = build_train(rng, args.condition, args.family, args.n_train, args.label_max, args.expand_max)
    train_keys = {P.key(p) for p in train}
    stats = {}
    if args.steps > 0:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                                 "gate_proj", "up_proj", "down_proj"],
                                                 task_type="CAUSAL_LM"))
        # LoRA weights in fp32 for stable fp16 training
        for n_, prm in model.named_parameters():
            if prm.requires_grad:
                prm.data = prm.data.float()
        enc = [encode_example(tok, p, args.max_len) for p in train]
        plens = np.array([e[2] for e in enc])
        stats = {"n_train": len(train), "prompt_tokens_mean": float(plens.mean()),
                 "prompt_tokens_p50_p90_max": [int(np.percentile(plens, 50)), int(np.percentile(plens, 90)), int(plens.max())],
                 "changes_mean": float(np.mean([p.n_changes() for p in train])),
                 "answer_max": max(p.answer() for p in train), "truncated": int(sum(len(e[0]) >= args.max_len for e in enc))}
        print(json.dumps({"train_stats": stats}), flush=True)
        opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.lr, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.0, 1 - s / args.steps))
        scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
        model.train()
        order = np.random.default_rng(args.seed).permutation(len(enc))
        for step in range(args.steps):
            idx = [order[(step * args.batch + j) % len(enc)] for j in range(args.batch)]
            ids, lab, att = (t.to(device) for t in collate([enc[i][:2] for i in idx], tok.pad_token_id))
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            if (step + 1) % max(args.steps // 10, 1) == 0:
                print(json.dumps({"step": step + 1, "loss": round(float(loss), 4), "elapsed": round(time.time() - t0)}), flush=True)
        model.eval()
        model.save_pretrained(os.path.join(args.out, "adapter"))
    train_s = time.time() - t0

    max_new = args.max_new or (16 if args.steps == 0 else 6)
    metrics, leak = {}, {}
    gen_path = open(os.path.join(args.out, "generations.jsonl"), "w")
    for tag, heldout, ks in (("train_vocab", False, args.eval_ks), ("heldout_vocab", True, [12])):
        sets = build_eval(args.seed, args.family, ks, args.eval_per_k, heldout)
        for k, probs in sets.items():
            leak[f"{tag}/{k}"] = sum(P.key(p) in train_keys for p in probs)
            outs = generate(model, tok, probs, device, max_new=max_new)
            preds = [parse(o) for o in outs]
            ok = [pr == p.answer() for pr, p in zip(preds, probs)]
            lenient = [parse_lenient(o) == p.answer() for o, p in zip(outs, probs)]
            metrics[f"{tag}/{k}"] = {"accuracy": float(np.mean(ok)), "accuracy_lenient": float(np.mean(lenient)),
                                     "parse_failures": int(sum(x is None for x in preds)), "n": len(probs)}
            for p, o, pr, c in zip(probs, outs, preds, ok):
                gen_path.write(json.dumps({"set": f"{tag}/{k}", "text": p.text(), "answer": p.answer(),
                                           "output": o, "parsed": pr, "correct": c}) + "\n")
            print(json.dumps({"eval": f"{tag}/{k}", **metrics[f"{tag}/{k}"]}), flush=True)
    gen_path.close()
    res = {"args": vars(args), "device": device, "torch": torch.__version__, "train_stats": stats,
           "leakage_exact_overlap": leak, "metrics": metrics, "train_seconds": round(train_s),
           "total_seconds": round(time.time() - t0),
           "generator_sha256": hashlib.sha256(open(P.__file__, "rb").read()).hexdigest(),
           "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()}
    json.dump(res, open(os.path.join(args.out, "metrics.json"), "w"), indent=1)
    print(json.dumps({"done": args.condition, "seed": args.seed, "metrics": metrics}), flush=True)


if __name__ == "__main__":
    main()
