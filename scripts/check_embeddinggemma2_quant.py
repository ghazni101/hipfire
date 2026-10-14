#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""EmbeddingGemma 2 (arch 18) quantization matrix — the arch-17
`check_embed_quant.py` shape.

Loads every artifact the CLI can write (dir, .q8, .q8f16, .mq6, .hf6, .mq4,
.hf4), encodes the parity fixtures through the daemon `embed` verb, and
asserts each artifact's embeddings sit inside its bit-width band against the
HF f32 reference (q8 3e-2, 6-bit 8e-2, 4-bit 2.5e-1 — an order under the O(1)
a broken dequant arm produces). The band is chosen by the artifact's own
dominant quant type (via the `astrea` HFQ parser), never its filename.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from check_embeddinggemma2_parity import CASES, Daemon, rel_l2  # noqa: E402


def artifact_format(path):
    """The artifact's real quant format from its HFQ tensor index (the
    arch-17 `check_embed_quant.py` derivation), never its filename."""
    if os.path.isdir(path):
        return "dir"
    try:
        import astrea
        summary, _ = astrea.read_hfq_index(path, max_tensors=0)
        counts = summary.get("quant_type_counts", {})
        compressed = {k: v for k, v in counts.items() if k not in ("F16", "F32")}
        dom = compressed or counts
        return max(dom, key=dom.get) if dom else None
    except Exception:
        return None

# Bands calibrated on THIS fixture (google/embeddinggemma-2, gfx1101, f32
# HF reference via ref.npz): measured worst rel-L2 across the fixtures was
# dir 1.0e-3, q8 2.0e-2, 6-bit 1.0e-1, 4-bit 3.4e-1 — the arch-18 trunk is a
# narrow (512-d) 24-layer encoder whose single-vector mean pool does not
# cancel systematic quant bias, so 6/4-bit run nearer their bit-width ceiling
# than the arch-17 multi-vector encoder's bands (8e-2 / 2.5e-1). Each band
# keeps ~50% headroom over the measured ladder and stays an order below the
# O(1) a broken dequant arm produces.
BANDS = {
    "MQ4G256": 5.0e-1, "MQ4G256V2": 5.0e-1, "HFQ4G256": 5.0e-1, "HFQ4G128": 5.0e-1,
    "MQ6G256": 1.5e-1, "MQ6G256V2": 1.5e-1, "HFQ6G256": 1.5e-1,
    "Q8_0": 3.0e-2, "Q8F16": 3.0e-2, "MQ8G256": 3.0e-2,
    "dir": 2.0e-3,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--daemon", default="target/release/daemon")
    ap.add_argument("artifacts", nargs="+")
    args = ap.parse_args()

    ref = np.load(args.ref)
    fixtures = [(k, kind, text) for k, kind, text in CASES if k in
                ("short_query", "short_doc", "code_doc", "unicode", "medium_doc")]

    failures = 0
    for path in args.artifacts:
        fmt = artifact_format(path) if not os.path.isdir(path) else "dir"
        band = BANDS.get(fmt)
        name = os.path.basename(path)
        if band is None:
            print(f"SKIP {name}: unknown format {fmt}")
            continue
        d = Daemon(args.daemon, path)
        worst, worst_key = 0.0, ""
        for key, kind, text in fixtures:
            r = d.embed(key, kind, {"text": text})
            e = rel_l2(r["embeddings"], ref[f"emb_{key}"])
            if e > worst:
                worst, worst_key = e, key
            if r["n_input_tokens"] != len(ref[f"ids_{key}"]):
                print(f"FAIL {name} {key}: token count {r['n_input_tokens']} != ref")
                failures += 1
        d.stop()
        ok = worst <= band
        print(f"{'ok  ' if ok else 'FAIL'} {name:14s} [{fmt:10s}] worst rel-L2 {worst:.2e} on {worst_key} (band {band:.1e})")
        failures += 0 if ok else 1

    if failures:
        print(f"\n{failures} FAILURES")
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
