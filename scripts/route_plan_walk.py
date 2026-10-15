#!/usr/bin/env -S python3 -I
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Record the kernel closure of one model route by driving the real dispatch.

One daemon, one model load, then a request walk over the route's axes (prompt
rows, decode context, greedy and sampled generates, session resumes, n-gram
copies, long context), with the developer-only HIPFIRE_MODULE_LOAD_AUDIT log.
`route_plan_build.py` turns one or more walk directories into the checked-in
route plan (crates/rdna-compute/src/route_plans/). Never used by product loading.

usage: route_plan_walk.py --run DIR --daemon BIN --model M --route ar|mtp
                          [--prompt-dir DIR] [--max-seq N] [--env K=V ...]
The environment (device selection, HIPFIRE_PLE_FORMAT, ...) is inherited; the
run gets its own HOME/HIPFIRE_HOME, lock dir and an empty kernel cache.
"""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import time

SAMPLING = [
    {"temperature": 0},
    {"temperature": 0.7, "top_p": 0.9},
    {"temperature": 0.7, "top_p": 0.9, "repeat_penalty": 1.1},
    {"temperature": 0.8, "top_k": 40, "seed": 7},
    {"temperature": 0.8, "min_p": 0.05},
    {"temperature": 0.6, "presence_penalty": 0.5, "frequency_penalty": 0.3},
    {"temperature": 0, "repeat_penalty": 1.2},
    {"temperature": 0, "top_logprobs": 5},
    {"temperature": 0.7, "logit_bias": {"1000": -5.0}},
]

PROMPTS = [
    "glimmer_coding_prompt.txt", "fiction_lighthouse.txt", "bare_factual.txt",
    "humaneval_0_has_close_elements.txt", "lru_cache_pep8_strict.txt", "code_edit_rewrite_copy.txt",
    "agentic_user_read.txt", "glimmer_prefill_1024.txt", "adaptive_kv_long_prefill.txt",
    "cohere2moe_long.txt",
]
FILLER = ("The lighthouse keeper logged the tide, the wind, the passing ships and the colour of the "
          "evening sky in a ledger that had outlived three keepers before him. ")


def prefill_rows():
    """Every row count to 300, then a stride with every 128-multiple +-2 to 4224, then the
    chunk, staging and context tails past it."""
    rows = set(range(1, 301))
    rows.update(range(301, 1101, 7))
    for base in range(128, 4225, 128):
        rows.update(base + d for d in (-2, -1, 0, 1, 2))
    rows.update([2047, 2048, 2049, 3000, 5000, 6143, 6144, 6145, 8191, 8192, 8193, 8200, 12289, 16385, 33000])
    return sorted(r for r in rows if r >= 1)


def walk_plan(route, prompt_dir, max_seq):
    plan = [{"type": "generate", "prompt": "Hi", "max_tokens": 16, "temperature": 0}]
    medium = (prompt_dir / "glimmer_coding_prompt.txt").read_text()
    for s in SAMPLING:
        plan.append(dict({"type": "generate", "prompt": medium, "max_tokens": 24}, **s))
    for name in PROMPTS:
        text = (prompt_dir / name).read_text()
        plan.append({"type": "generate", "prompt": text, "max_tokens": 96, "temperature": 0, "_name": name})
    for name in ("code_edit_rewrite_copy.txt", "glimmer_coding_prompt.txt", "glimmer_prefill_1024.txt"):
        text = (prompt_dir / name).read_text()
        plan.append({"type": "generate", "prompt": text, "max_tokens": 96, "temperature": 0.7, "top_p": 0.9,
                     "_name": name})
    # n-gram takeover: long verbatim copies, greedy and sampled.
    copy = "Repeat the following text exactly, twice:\n\n" + FILLER * 12
    plan.append({"type": "generate", "prompt": copy, "max_tokens": 192, "temperature": 0})
    plan.append({"type": "generate", "prompt": copy, "max_tokens": 192, "temperature": 0.6, "top_p": 0.95})
    # Session resumes (append prefill onto a cached prefix), greedy then sampled.
    plan.append({"type": "generate", "prompt": "Name three rivers in Europe.", "max_tokens": 40, "temperature": 0})
    plan.append({"type": "generate", "chain_user": "Now name two more.", "max_tokens": 40, "temperature": 0})
    plan.append({"type": "generate", "chain_user": "Which is longest?", "max_tokens": 40, "temperature": 0.7,
                 "top_p": 0.9})
    # Long context (QSA selection past the dense window): ~20.7K-token code and ~31K synthetic.
    if max_seq > 34000:
        code = (prompt_dir / "qwen38_issue693_longcode_20676.txt").read_text()
        plan.append({"type": "generate", "prompt": code, "max_tokens": 32, "temperature": 0, "_name": "20k"})
        plan.append({"type": "generate", "prompt": code, "max_tokens": 32, "temperature": 0.7, "top_p": 0.9,
                     "_name": "20k-sampled"})
        plan.append({"type": "generate", "prompt": "Summarize this ledger.\n\n" + FILLER * 900,
                     "max_tokens": 32, "temperature": 0, "_name": "31k"})
    if max_seq > 110000:
        plan.append({"type": "generate", "prompt": "Summarize this ledger.\n\n" + FILLER * 2900,
                     "max_tokens": 24, "temperature": 0.7, "top_p": 0.9, "_name": "100k-sampled"})
    if route == "ar":
        for rows in prefill_rows():
            if rows < max_seq:
                plan.append({"type": "bench_prefill", "tokens": rows})
        for ctx in (1, 2, 127, 128, 1000, 2047, 2048, 4095, 4096, 8192, 9000, 16400, 33000, 65536, 120000):
            if ctx + 8 < max_seq:
                plan.append({"type": "bench_decode", "context_tokens": ctx, "iterations": 3})
    else:
        # Native-MTP prompt fill runs on generate only: sweep prompt lengths (rows include the
        # chat template) with short decodes, plus the chunk tails.
        words = FILLER.split()
        for n in list(range(1, 65)) + list(range(66, 1100, 9)) + [2040, 2050, 4090, 4100, 4200, 8190, 8200]:
            text = " ".join((words * (n // len(words) + 1))[:n])
            plan.append({"type": "generate", "prompt": text, "max_tokens": 6, "temperature": 0, "_words": n})
        for n in (40, 300, 900):
            text = " ".join((words * (n // len(words) + 1))[:n])
            plan.append({"type": "generate", "prompt": text, "max_tokens": 12, "temperature": 0.8,
                         "top_k": 40, "seed": n, "_words": n})
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--daemon", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--route", choices=["ar", "mtp"], required=True)
    ap.add_argument("--prompt-dir", default=str(pathlib.Path(__file__).resolve().parent.parent
                                                / "benchmarks" / "prompts"))
    ap.add_argument("--max-seq", type=int, default=40960)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--timeout", type=float, default=3000)
    ap.add_argument("--sealed", action="store_true",
                    help="validate a checked-in plan: keep route plans on, so any kernel the sealed "
                         "route did not plan fails its request")
    a = ap.parse_args()

    run = pathlib.Path(a.run)
    run.mkdir(parents=True, exist_ok=False)
    for name in ("home/.hipfire", "lockdir", "kcache"):
        (run / name).mkdir(parents=True)
    plan = walk_plan(a.route, pathlib.Path(a.prompt_dir), a.max_seq)
    env = os.environ.copy()
    env.update(HOME=str(run / "home"), HIPFIRE_HOME=str(run / "home" / ".hipfire"),
               HIPFIRE_LOCK_DIR=str(run / "lockdir"), HIPFIRE_KERNEL_CACHE=str(run / "kcache"),
               HIPFIRE_MODULE_LOAD_AUDIT="1")
    if not a.sealed:
        # Observe the lazy closure, not a checked-in plan's preload.
        env["HIPFIRE_ROUTE_PLAN"] = "0"
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    load = {"type": "load", "model": a.model,
            "params": {"max_seq": a.max_seq, "mtp_mode": "on" if a.route == "mtp" else "off"}}
    daemon = pathlib.Path(a.daemon)
    (run / "command.json").write_text(json.dumps({
        "daemon": str(daemon), "daemon_sha256": hashlib.sha256(daemon.read_bytes()).hexdigest(),
        "route": a.route, "env": {k: v for k, v in env.items() if k.startswith(("HIP", "ROCR")) or k == "HOME"},
        "load": load, "requests": len(plan)}, indent=1))
    (run / "plan.json").write_text(json.dumps(plan, indent=1))

    err = (run / "stderr.log").open("w")
    out = (run / "stdout.log").open("w")
    proc = subprocess.Popen([str(daemon)], env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=err, text=True, bufsize=1)
    deadline = time.monotonic() + a.timeout
    last_text = ""

    def send(msg, done_types):
        nonlocal last_text
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()
        t = time.monotonic()
        last_text = ""
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if not line:
                raise SystemExit(f"daemon exited during {msg.get('type')}")
            out.write(line)
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if obj.get("type") == "token":
                last_text += obj.get("text", "")
            if obj.get("type") == "commit_ready":
                proc.stdin.write(json.dumps({"type": "commit", "id": obj["id"],
                                             "attempt_id": obj["attempt_id"]}) + "\n")
                proc.stdin.flush()
            if obj.get("type") in done_types:
                return obj, time.monotonic() - t
        raise SystemExit("walk timeout")

    results, errors = [], 0
    obj, dt = send(load, ("loaded", "error"))
    results.append({"req": "load", "resp": obj, "s": round(dt, 2)})
    print(json.dumps({"load": obj.get("type"), "s": round(dt, 2)}), flush=True)
    if obj.get("type") == "loaded":
        history = []
        for i, msg in enumerate(plan):
            msg = {k: v for k, v in msg.items() if not k.startswith("_")}
            if msg["type"] == "generate":
                msg.setdefault("id", f"g{i}")
                msg.setdefault("attempt_id", i + 1)
                msg.setdefault("max_think_tokens", 1)
                if "chain_user" in msg:
                    history = history + [{"role": "assistant", "content": last_text},
                                         {"role": "user", "content": msg.pop("chain_user")}]
                    msg["messages"] = history
                else:
                    history = [{"role": "user", "content": msg["prompt"]}]
                done = ("done", "error")
            elif msg["type"] == "bench_prefill":
                done = ("prefill_result", "error")
            else:
                done = ("decode_result", "bench_decode_result", "error", "done")
            # Request boundary in the audit stream.
            print(f"ROUTE_WALK_REQ\t{i}\t{json.dumps({k: v for k, v in msg.items() if k not in ('prompt', 'messages')})}",
                  file=err, flush=True)
            obj, dt = send(msg, done)
            results.append({"i": i, "type": msg["type"], "resp_type": obj.get("type"), "s": round(dt, 3),
                            "error": obj if obj.get("type") == "error" else None})
            if obj.get("type") == "error":
                errors += 1
                print(json.dumps({"i": i, "error": obj})[:400], flush=True)
    try:
        proc.stdin.write(json.dumps({"type": "unload"}) + "\n")
        proc.stdin.close()
    except BrokenPipeError:
        pass
    rc = proc.wait(timeout=180)
    ok = obj.get("type") != "error" and rc == 0 and errors == 0 and len(results) == len(plan) + 1
    (run / "result.json").write_text(json.dumps({"rc": rc, "ok": ok, "errors": errors, "results": results}, indent=1))
    print(json.dumps({"rc": rc, "ok": ok, "errors": errors, "requests": len(results) - 1}), flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
