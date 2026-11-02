import re, sys
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), ".."))
import benchmarks as bm
RESULT = re.compile(r"=\s*\$?\s*(-?(?:\\[dt]?frac\{-?\d+\}\{\d+\}|\d+(?:\.\d+)?(?:\s*/\s*\d+)?))(?!\s*[\d.,]*\d)(?!\s*[-+*/^_(\\a-zA-Z{])")
def results(t):
    out = set()
    for m in RESULT.finditer(t):
        v = bm.normalize(m.group(1))
        if v: out.add(v)
    return out
