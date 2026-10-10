"""Small coding eval: does the model's code pass hidden unit tests?

usage: uv run python scripts/code_eval.py MODEL [--think] [--only N,N,...] [--ctx 4096]

Each problem names a function/class; the model's code block is executed in a
subprocess together with the asserts (10s timeout). Reports pass rate,
wall time, generated tokens and generation speed.
"""
import json, re, subprocess, sys, tempfile, time
import httpx

URL = "http://localhost:11434"

PROBLEMS = [
    ("int_to_roman", "Write `int_to_roman(n: int) -> str` converting 1..3999 to Roman numerals.",
     "assert int_to_roman(1994)=='MCMXCIV'\nassert int_to_roman(3999)=='MMMCMXCIX'\nassert int_to_roman(4)=='IV'\nassert int_to_roman(58)=='LVIII'"),
    ("merge_intervals", "Write `merge_intervals(iv: list[list[int]]) -> list[list[int]]` merging overlapping or touching intervals, result sorted by start.",
     "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]])==[[1,6],[8,10],[15,18]]\nassert merge_intervals([[1,4],[4,5]])==[[1,5]]\nassert merge_intervals([])==[]\nassert merge_intervals([[5,7],[1,2]])==[[1,2],[5,7]]"),
    ("longest_palindrome", "Write `longest_palindrome(s: str) -> str` returning the longest palindromic substring (the leftmost one if tied).",
     "assert longest_palindrome('babad')=='bab'\nassert longest_palindrome('cbbd')=='bb'\nassert longest_palindrome('a')=='a'\nassert longest_palindrome('forgeeksskeegfor')=='geeksskeeg'\nassert longest_palindrome('')==''"),
    ("LRUCache", "Write a class `LRUCache(capacity: int)` with `get(key) -> int` (returns -1 if missing) and `put(key, value)`, evicting the least recently used key when full. Both O(1).",
     "c=LRUCache(2);c.put(1,1);c.put(2,2);assert c.get(1)==1;c.put(3,3);assert c.get(2)==-1;c.put(4,4);assert c.get(1)==-1;assert c.get(3)==3;assert c.get(4)==4\nc=LRUCache(1);c.put(1,1);c.put(1,2);assert c.get(1)==2"),
    ("eval_rpn", "Write `eval_rpn(tokens: list[str]) -> int` evaluating Reverse Polish Notation with + - * /, where division truncates toward zero.",
     "assert eval_rpn(['2','1','+','3','*'])==9\nassert eval_rpn(['4','13','5','/','+'])==6\nassert eval_rpn(['10','6','9','3','+','-11','*','/','*','17','+','5','+'])==22\nassert eval_rpn(['7','-3','/'])==-2"),
    ("edit_distance", "Write `edit_distance(a: str, b: str) -> int`, the Levenshtein distance.",
     "assert edit_distance('horse','ros')==3\nassert edit_distance('intention','execution')==5\nassert edit_distance('','abc')==3\nassert edit_distance('same','same')==0"),
    ("top_k_words", "Write `top_k_words(words: list[str], k: int) -> list[str]` returning the k most frequent words, ordered by frequency descending, ties broken alphabetically.",
     "assert top_k_words(['i','love','leetcode','i','love','coding'],2)==['i','love']\nassert top_k_words(['the','day','is','sunny','the','the','the','sunny','is','is'],4)==['the','is','sunny','day']\nassert top_k_words(['b','a','c'],2)==['a','b']"),
    ("parse_duration", "Write `parse_duration(s: str) -> int` that converts strings like '1h30m', '45s', '2h5s', '1d2h' (units d,h,m,s, each at most once, in that order) into total seconds. Raise ValueError for anything invalid, including the empty string, unknown units, repeated or out-of-order units.",
     "assert parse_duration('1h30m')==5400\nassert parse_duration('45s')==45\nassert parse_duration('2h5s')==7205\nassert parse_duration('1d2h')==93600\nfor bad in ['','5x','1m1h','1h1h','h','1h 2m','-1s']:\n    try:\n        parse_duration(bad)\n    except ValueError:\n        pass\n    else:\n        raise AssertionError(bad)"),
    ("topo_sort", "Write `topo_sort(deps: dict[str, list[str]]) -> list[str]` where deps maps a node to the nodes it depends on (which must come before it). Nodes may appear only as dependencies. Return an order where, among nodes that are ready at the same time, the alphabetically smallest comes first. Raise ValueError on a cycle.",
     "assert topo_sort({'b':['a'],'c':['a','b']})==['a','b','c']\nassert topo_sort({'x':[],'a':[]})==['a','x']\nassert topo_sort({'d':['b','c'],'b':['a'],'c':['a']})==['a','b','c','d']\ntry:\n    topo_sort({'a':['b'],'b':['a']})\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('cycle')"),
    ("flatten_dict", "Write `flatten_dict(d: dict, sep: str = '.') -> dict` flattening nested dicts into keys joined by sep. Empty nested dicts are dropped. Non-dict values (including lists) are kept as-is.",
     "assert flatten_dict({'a':1,'b':{'c':2,'d':{'e':3}}})=={'a':1,'b.c':2,'b.d.e':3}\nassert flatten_dict({'a':{}, 'b':[1,{'x':1}]})=={'b':[1,{'x':1}]}\nassert flatten_dict({'a':{'b':1}},sep='/')=={'a/b':1}"),
    ("word_break_all", "Write `word_break_all(s: str, words: list[str]) -> list[str]` returning every way to split s into dictionary words, each as a space-joined sentence, sorted alphabetically.",
     "assert word_break_all('catsanddog',['cat','cats','and','sand','dog'])==['cat sand dog','cats and dog']\nassert word_break_all('pineapplepenapple',['apple','pen','applepen','pine','pineapple'])==['pine apple pen apple','pine applepen apple','pineapple pen apple']\nassert word_break_all('catsandog',['cats','dog','sand','and','cat'])==[]"),
    ("spiral", "Write `spiral(matrix: list[list[int]]) -> list[int]` returning elements in clockwise spiral order.",
     "assert spiral([[1,2,3],[4,5,6],[7,8,9]])==[1,2,3,6,9,8,7,4,5]\nassert spiral([[1,2,3,4],[5,6,7,8],[9,10,11,12]])==[1,2,3,4,8,12,11,10,9,5,6,7]\nassert spiral([])==[]\nassert spiral([[1],[2],[3]])==[1,2,3]"),
]

SYSTEM = "You are an expert Python programmer. Reply with a single ```python code block containing the complete solution and nothing else."


def extract(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return max(blocks, key=len) if blocks else text


def run_tests(code, tests):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code + "\n\n" + tests + "\nprint('PASS')\n")
    try:
        p = subprocess.run([sys.executable, "-I", f.name], capture_output=True, text=True, timeout=10,
                           cwd=tempfile.gettempdir())
        return "PASS" in p.stdout
    except subprocess.TimeoutExpired:
        return False


def main():
    args = sys.argv[1:]
    model = args[0]
    think = "--think" in args
    ctx = int(args[args.index("--ctx") + 1]) if "--ctx" in args else 4096
    only = None
    if "--only" in args:
        only = {int(x) for x in args[args.index("--only") + 1].split(",")}
    num_predict = 4096 if think else 1024

    passed = total = 0
    tokens = gen_s = wall = 0.0
    with httpx.Client(timeout=1800) as c:
        for i, (name, prompt, tests) in enumerate(PROBLEMS, 1):
            if only and i not in only:
                continue
            t0 = time.perf_counter()
            r = c.post(f"{URL}/api/chat", json={
                "model": model, "stream": False, "think": think,
                "options": {"num_ctx": ctx, "num_predict": num_predict,
                            "temperature": 0.6 if think else 0.7, "top_p": 0.95 if think else 0.8, "top_k": 20},
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            }).json()
            dt = time.perf_counter() - t0
            ok = run_tests(extract(r["message"].get("content", "")), tests)
            total += 1; passed += ok; wall += dt
            tokens += r.get("eval_count", 0); gen_s += r.get("eval_duration", 0) / 1e9
            print(f"  {i:2d} {name:20s} {'PASS' if ok else 'fail'}  {dt:6.1f}s  {r.get('eval_count', 0):5d} tok", flush=True)
        ps = c.get(f"{URL}/api/ps").json()["models"]
        gpu = next((m["size_vram"] / m["size"] * 100 for m in ps if m["name"].startswith(model.split(":")[0])), 0)

    summary = {"model": model, "think": think, "passed": passed, "total": total,
               "avg_s": round(wall / total, 1), "tok_s": round(tokens / gen_s, 1) if gen_s else 0,
               "gpu_pct": round(gpu)}
    print("RESULT " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
