"""G1 probe: how fast does short-only training learn, and how does it do on longer expressions?"""
import sys, time, pickle
import numpy as np
import lab

seed, steps, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
L = lab.Learner(seed, lr=1e-3, d=64, layers=3, heads=4, mlp=256)
rng = np.random.default_rng(seed)
s = lab.labeled_sampler(rng, [1, 2, 3, 4], batch=128)
t = time.time()
for k in range(steps // 1000):
    L.train(s, 1000)
    print("steps", 1000 * (k + 1), lab.evaluate(L, np.random.default_rng(999), range(1, 9), 300), f"{time.time()-t:.0f}s", flush=True)
pickle.dump(L.params, open(out, "wb"))
