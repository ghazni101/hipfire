#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""EmbeddingGemma 2 (arch 18) multi-slot route check — the arch-17
`check_embed_multislot.py` shape.

Encodes the same inputs through the ordinary daemon (sequential) and through
`experimental_multi_slot` (the arch-18 slot engine's batched forward) and
requires them **bit-identical** — the batched forward is deterministic
(nothing carries state across segments), so any difference is a segment
cross-read or a batch-layout bug, exactly the failure class this route can
have. Also diffs the slot route against the HF f32 reference (rel-L2 band
2e-2) so the batch path is anchored to ground truth, not just self-agreement.
"""
import queue
import subprocess
import sys
import threading

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from check_embeddinggemma2_parity import CASES, Daemon, rel_l2  # noqa: E402


def embed_concurrent(d, tag, kind, text, out):
    out[tag] = d.embed(tag, kind, {"text": text})


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--daemon", default="target/release/daemon")
    ap.add_argument("--slots", type=int, default=4)
    args = ap.parse_args()

    ref = np.load(args.ref)
    bodies = [(k, kind, text) for k, kind, text in CASES if k != "long_doc"]

    seq = Daemon(args.daemon, args.model)
    print(f"sequential daemon: arch {seq.loaded.get('arch')}")
    seq_out = {}
    for i, (key, kind, text) in enumerate(bodies):
        seq_out[key] = seq.embed(f"seq{i}", kind, {"text": text})
    seq.stop()

    slot = Daemon(args.daemon, args.model, params={
        "experimental_multi_slot": True,
        "experimental_multi_slot_slots": args.slots,
        "max_batch_tokens": 8192,
        "max_seq": 8192,
    })
    if not slot.loaded.get("experimental_multi_slot"):
        print("FAIL: daemon did not enter experimental_multi_slot")
        return 1
    print(f"slot daemon: arch {slot.loaded.get('arch')} (experimental_multi_slot, {args.slots} slots)")

    failures = 0
    # All requests in flight at once — the collector must merge them into one
    # row block (mixed lengths force per-segment attention and per-segment
    # position restarts inside one forward).
    threads, slot_out, errs = [], {}, {}
    for i, (key, kind, text) in enumerate(bodies):
        t = threading.Thread(
            target=embed_concurrent, args=(slot, f"slot{i}", kind, text, slot_out), daemon=True
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    for i, (key, kind, text) in enumerate(bodies):
        try:
            r = slot_out[f"slot{i}"]
        except RuntimeError as e:
            print(f"FAIL {key}: slot route error: {e}")
            failures += 1
            continue
        s = seq_out[key]
        same = np.array_equal(
            np.array(r["embeddings"], dtype=np.float32),
            np.array(s["embeddings"], dtype=np.float32),
        )
        e = rel_l2(r["embeddings"], ref[f"emb_{key}"])
        ok = same and e <= 2e-2 and r["n_input_tokens"] == s["n_input_tokens"]
        print(
            f"{'ok  ' if ok else 'FAIL'} {key}: bit-identical={same} slot-vs-ref rel-L2 {e:.2e}"
        )
        failures += 0 if ok else 1
    slot.stop()

    if failures:
        print(f"\n{failures} FAILURES")
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
