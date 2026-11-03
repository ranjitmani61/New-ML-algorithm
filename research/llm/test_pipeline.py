"""Pipeline checks for run_llm.py with a tiny random Qwen2 model (CPU, no downloads).

    python3 test_pipeline.py TINY_MODEL_DIR

1. Loss masking: the only tokens that carry loss decode to exactly "<answer><eos>".
2. Format consistency: after overfitting 16 problems, generating on the same prompts reproduces
   their answers. A mismatch between the training format and the evaluation prompt would fail.
"""

import random
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import problems as P
import run_llm as R

model_dir = sys.argv[1]
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
rng = random.Random(0)
probs = [P.make_problem(rng, rng.randint(1, 4)) for _ in range(16)]
enc = [R.encode_example(tok, p, 384) for p in probs]

for p, (ids, labels, plen) in zip(probs, enc):
    kept = [i for i, l in zip(ids, labels) if l != -100]
    assert kept == [l for l in labels if l != -100]
    want = tok(str(p.answer()), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    assert kept == want, (tok.convert_ids_to_tokens(kept), p.answer())
    assert tok.decode(kept, skip_special_tokens=True) == str(p.answer())
    assert all(l == -100 for l in labels[:plen])
print("1. loss masking: only the answer and end-of-turn tokens carry loss (16/16)")

opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
model.train()
for step in range(400):
    ids, lab, att = R.collate([e[:2] for e in enc], tok.pad_token_id)
    loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
    loss.backward(); opt.step(); opt.zero_grad()
model.eval()
outs = R.generate(model, tok, probs, "cpu", batch=8, max_new=6)
preds = [R.parse(o) for o in outs]
acc = float(np.mean([pr == p.answer() for pr, p in zip(preds, probs)]))
print(f"2. overfit 16 problems: final loss {loss.item():.4f}, exact-match on the same prompts {acc:.2f}")
for p, o in list(zip(probs, outs))[:3]:
    print("   ", repr(p.text()[-60:]), "->", repr(o), "answer", p.answer())
assert acc >= 0.9, "training format and evaluation prompt do not match"
print("pipeline checks passed")
