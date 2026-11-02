"""Dump cluster keys, gold key and per-question grading details for determinism and bias checks."""
import sys, pickle, time, io, contextlib
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), ".."))
import grading
t = time.time()
buf = io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    qs = grading.load_pools(sys.argv[1], {sys.argv[2]})
out = {q.qid: ([t.answer for t in q.traces], q.gold, [t.mentions for t in q.traces], q.problem_values) for q in qs}
pickle.dump({"out": out, "log": buf.getvalue(), "seconds": time.time() - t}, open(sys.argv[3], "wb"))
print(sys.argv[3], len(out), "timeouts", buf.getvalue().count("Timeout"), round(time.time() - t), "s")
