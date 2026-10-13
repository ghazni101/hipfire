#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""EmbeddingGemma-2 (arch 18) end-to-end parity check — the oracle leg.

Drives the daemon's `embed` verb over a safetensors dir and compares the
served 768-d embeddings (and their Matryoshka truncations) against a
float32 CPU `transformers` reference produced by
`make_reference.py` (google/embeddinggemma-2, `AutoModel` + mean pool +
L2), plus the edge-case contract: image input refused, empty text refused,
oversized and non-integer `dims` refused, prompt-exceeds-limit refused, bad
kind refused, and >8192-token input truncated to exactly 8192.

Expect hipfire (f16 GEMM weights, f32 activations) at ~1e-3 relative-L2
against the f32 reference; the band is 2e-2 (two orders of margin, still
tight enough that a misread convention — wrong norm order, missing PLE,
wrong window — fails by O(1)).

Usage:
    python3 scripts/check_embeddinggemma2_parity.py \
        --model <dir-with-model.safetensors> --ref <ref.npz> \
        [--daemon target/release/daemon]
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading

import numpy as np

Q = "task: search result | query: "
D = "title: none | text: "

# (ref.npz key, kind, prompt+text) — texts byte-identical to make_reference.py.
CASES = [
    ("short_query", "query", "what is hipfire"),
    ("short_doc", "document", "hipfire is an inference engine for AMD GPUs."),
    ("code_doc", "document", "def quicksort(a):\n    if len(a) <= 1:\n        return a\n    return quicksort([x for x in a[1:] if x < a[0]]) + [a[0]] + quicksort([x for x in a[1:] if x >= a[0]])"),
    ("unicode", "document", "Straße café naïve 日本語のテキストです。🚀 ünïcödé"),
    ("emoji", "query", "fire rocket 🚀🔥 lol"),
    ("one_char", "document", "x"),
    ("medium_doc", "document", ("The quick brown fox jumps over the lazy dog. " * 40).strip()),
    # Past the 512-radius window and NOT 64-aligned: the eager path runs with
    # both the sliding band mask and padding rows in play.
    ("window_pad_doc", "document", ("The quick brown fox jumps over the lazy dog. " * 90).strip()),
    ("long_doc", "document", ("The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. " * 1200).strip()),
]

MRL = [512, 256, 128]
BAND = 2e-2


class Daemon:
    def __init__(self, binary, model, params=None):
        self.proc = subprocess.Popen(
            [binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        self._stderr_buf = []
        threading.Thread(target=self._drain, daemon=True).start()
        self._wlock = threading.Lock()
        self._pending = {}
        self._plock = threading.Lock()
        threading.Thread(target=self._read, daemon=True).start()
        self.send({"type": "load", "model": model, "params": params or {}})
        # The `loaded` reply to a load carries no id — match on None.
        self.loaded = self._wait("loaded", None)

    def _drain(self):
        for line in self.proc.stderr:
            self._stderr_buf.append(line)

    def _read(self):
        for line in self.proc.stdout:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            with self._plock:
                q = self._pending.get(d.get("id"))
            if q is not None:
                q.put(d)

    def send(self, obj):
        with self._wlock:
            self.proc.stdin.write(json.dumps(obj) + "\n")
            self.proc.stdin.flush()

    def _wait(self, want, rid, timeout=900.0):
        q = queue.Queue()
        with self._plock:
            self._pending[rid] = q
        try:
            d = q.get(timeout=timeout)
        except queue.Empty:
            # A second daemon cannot load while another holds the GPU lock, so
            # "no reply" usually means the child died — surface its stderr
            # rather than a bare timeout.
            tail = "".join(self._stderr_buf[-6:])
            raise RuntimeError(
                f"daemon gave no {want!r} reply in {timeout:.0f}s — stderr tail:\n{tail}"
            )
        finally:
            with self._plock:
                self._pending.pop(rid, None)
        if d.get("type") == "error":
            raise RuntimeError(f"daemon error: {d.get('message')}")
        return d

    def embed(self, rid, kind, input_obj, dims=None, expect_error=False):
        req = {"type": "embed", "id": str(rid), "kind": kind, "input": input_obj}
        if dims is not None:
            req["dims"] = dims
        d = self._wait_embed(req, str(rid))
        if expect_error:
            if d.get("type") != "error":
                raise AssertionError(f"expected an error, got {d.get('type')}: {str(d)[:200]}")
            return d.get("message", "")
        if d.get("type") == "error":
            raise RuntimeError(f"embed {rid}: {d.get('message')}")
        return d

    def _wait_embed(self, req, rid):
        q = queue.Queue()
        with self._plock:
            self._pending[rid] = q
        try:
            self.send(req)
            return q.get(timeout=900.0)
        finally:
            with self._plock:
                self._pending.pop(rid, None)

    def stop(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def rel_l2(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(b)
    if denom == 0:
        return float(np.linalg.norm(a - b))
    return float(np.linalg.norm(a - b) / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--daemon", default="target/release/daemon")
    args = ap.parse_args()

    ref = np.load(args.ref)
    d = Daemon(args.daemon, args.model)
    print(f"loaded: arch={d.loaded.get('arch')}")

    failures = 0
    rid = 0
    for key, kind, text in CASES:
        rid += 1
        r = d.embed(rid, kind, {"text": text})
        ref_ids = ref[f"ids_{key}"]
        if r["n_input_tokens"] != len(ref_ids):
            print(f"FAIL {key}: n_input_tokens {r['n_input_tokens']} != ref {len(ref_ids)}")
            failures += 1
            continue
        got = np.array(r["embeddings"], dtype=np.float32)
        e = rel_l2(got, ref[f"emb_{key}"])
        ok = e <= BAND
        print(f"{'ok  ' if ok else 'FAIL'} {key}: n={r['n_input_tokens']} dim={r['dim']} rel-L2 {e:.2e}")
        failures += 0 if ok else 1
        # Matryoshka truncations must reproduce the reference's truncated
        # re-normalized vectors.
        for m in MRL:
            rid += 1
            r2 = d.embed(rid, kind, {"text": text}, dims=m)
            got2 = np.array(r2["embeddings"], dtype=np.float32)
            e2 = rel_l2(got2, ref[f"emb_{key}_{m}"])
            ok2 = e2 <= BAND and r2["dim"] == m and len(got2) == m
            print(f"{'ok  ' if ok2 else 'FAIL'}   mrl {m}: rel-L2 {e2:.2e} dim {r2['dim']}")
            failures += 0 if ok2 else 1

    # ── edge cases ────────────────────────────────────────────────────
    rid += 1
    msg = d.embed(rid, "query", {"text": "a cat", "image_base64": "iVBORw0KGgo="}, expect_error=True)
    ok = "text-only" in msg
    print(f"{'ok  ' if ok else 'FAIL'} image input refused: {msg[:90]}")
    failures += 0 if ok else 1

    rid += 1
    msg = d.embed(rid, "query", {"text": ""}, expect_error=True)
    ok = bool(msg)
    print(f"{'ok  ' if ok else 'FAIL'} empty text refused: {msg[:90]}")
    failures += 0 if ok else 1

    rid += 1
    msg = d.embed(rid, "query", {"text": "hello"}, dims=999, expect_error=True)
    ok = "999" in msg
    print(f"{'ok  ' if ok else 'FAIL'} oversized dims refused: {msg[:90]}")
    failures += 0 if ok else 1

    # kind is validated by parse_kind — drive it through a raw request:
    d.send({"type": "embed", "id": "badkind", "kind": "poem", "input": "x"})
    try:
        resp = d._wait(None, "badkind")
        ok = False
        detail = str(resp)[:90]
    except RuntimeError as e:
        ok = True
        detail = str(e)[:90]
    print(f"{'ok  ' if ok else 'FAIL'} bad kind refused: {detail}")
    failures += 0 if ok else 1

    # Long doc truncation: the reference was tokenized with HF truncation at
    # 8192; hipfire must land on exactly 8192 input tokens.
    rid += 1
    r = d.embed(rid, "document", {"text": CASES[-1][2]})
    ok = r["n_input_tokens"] == len(ref["ids_long_doc"]) == 8192
    print(f"{'ok  ' if ok else 'FAIL'} long doc truncated to {r['n_input_tokens']} (ref {len(ref['ids_long_doc'])})")
    failures += 0 if ok else 1

    # dims must be a strictly non-negative integer: a float/string silently
    # meaning "full width" is a protocol trap, not a convenience.
    rid += 1
    msg = d.embed(rid, "query", {"text": "hello"}, dims=12.5, expect_error=True)
    ok = "non-negative integer" in msg
    print(f"{'ok  ' if ok else 'FAIL'} non-integer dims refused: {msg[:90]}")
    failures += 0 if ok else 1

    d.stop()

    # Prompt-exceeds-limit: truncation makes any *body* fit, so the guard runs
    # on the kind prompt's own tokenization. The shipped checkpoint declares no
    # length keys, so the cheapest honest fixture is a symlink farm over the
    # same weights with a 4-token query limit. (`d` is stopped first: a second
    # daemon cannot load while another holds the GPU lock.)
    with tempfile.TemporaryDirectory() as tiny:
        for entry in os.listdir(args.model):
            if entry == "config_sentence_transformers.json":
                continue  # written fresh below (a symlink would write through)
            os.symlink(os.path.join(args.model, entry), os.path.join(tiny, entry))
        with open(os.path.join(tiny, "config_sentence_transformers.json"), "w") as fh:
            json.dump(
                {"prompts": {"query": Q, "document": D},
                 "query_length": 4, "document_length": 8192},
                fh,
            )
        tiny_d = Daemon(args.daemon, tiny)
        msg = tiny_d.embed("tinyq", "query", {"text": "hello world"}, expect_error=True)
        ok = "leaving no room" in msg
        print(f"{'ok  ' if ok else 'FAIL'} prompt-exceeds-limit refused: {msg[:90]}")
        failures += 0 if ok else 1
        r = tiny_d.embed("tinyd", "document", {"text": "hello world"})
        ok = r.get("type") == "embedded"
        print(f"{'ok  ' if ok else 'FAIL'} document under its own 8192 limit still encodes")
        failures += 0 if ok else 1
        tiny_d.stop()

    if failures:
        print(f"\n{failures} FAILURES")
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
