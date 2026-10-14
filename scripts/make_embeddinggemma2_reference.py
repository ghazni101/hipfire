#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""EmbeddingGemma-2 (arch 18) parity reference generator — the `ref.npz`
`check_embeddinggemma2_parity.py` compares against.

float32 CPU `transformers`: `AutoModel` (google/embeddinggemma-2) →
last_hidden_state (the checkpoint's own 768-d `embedding_projection`)
→ sentence-transformers mean pooling over every token (the prompts are
pooled too, `1_Pooling` `include_prompt: true`) → L2 → Matryoshka
truncations 512/256/128 re-normalized. Texts and kind prompts are
byte-identical to the parity script's CASES; HF `tokenizers` truncation
at 8192 (special-token-aware, bos/eos frame survives) matches
`assemble_input`'s body-only cut, so `ids_<key>` asserts the wire path's
token count bit-for-bit.

Usage:
    python3 scripts/make_embeddinggemma2_reference.py \
        --model <dir-with-model.safetensors> --out ref.npz

Needs torch (CPU suffices) + transformers >= 5.18 (`embedding_gemma2`
architecture) + numpy.
"""
import argparse
import glob
import os

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

Q = "task: search result | query: "
D = "title: none | text: "

# Byte-identical to check_embeddinggemma2_parity.py CASES — keep the two in
# lockstep; the parity script indexes this file by key.
CASES = [
    ("short_query", "query", "what is hipfire"),
    ("short_doc", "document", "hipfire is an inference engine for AMD GPUs."),
    ("code_doc", "document", "def quicksort(a):\n    if len(a) <= 1:\n        return a\n    return quicksort([x for x in a[1:] if x < a[0]]) + [a[0]] + quicksort([x for x in a[1:] if x >= a[0]])"),
    ("unicode", "document", "Straße café naïve 日本語のテキストです。🚀 ünïcödé"),
    ("emoji", "query", "fire rocket 🚀🔥 lol"),
    ("one_char", "document", "x"),
    ("medium_doc", "document", ("The quick brown fox jumps over the lazy dog. " * 40).strip()),
    ("window_pad_doc", "document", ("The quick brown fox jumps over the lazy dog. " * 90).strip()),
    ("long_doc", "document", ("The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. " * 1200).strip()),
]

MRL = [512, 256, 128]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="dir with model.safetensors (HF snapshot or a glob of it)")
    ap.add_argument("--out", required=True, help="output .npz path")
    args = ap.parse_args()
    model_dir = args.model
    if os.path.isdir(model_dir) and not os.path.exists(
            os.path.join(model_dir, "model.safetensors")):
        hits = sorted(glob.glob(os.path.join(model_dir, "*", "model.safetensors")))
        if hits:
            model_dir = os.path.dirname(hits[0])

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="eager")
    model.eval()

    out = {}
    with torch.no_grad():
        for key, kind, text in CASES:
            prompt = Q if kind == "query" else D
            enc = tok(prompt + text, return_tensors="pt",
                      truncation=True, max_length=8192)
            ids = enc["input_ids"][0]
            out[f"ids_{key}"] = ids.numpy()
            hs = model(**enc).last_hidden_state[0]  # [n, 768], already projected
            pooled = hs.mean(dim=0)
            pooled = pooled / pooled.norm()
            out[f"emb_{key}"] = pooled.numpy()
            for m in MRL:
                t = pooled[:m]
                out[f"emb_{key}_{m}"] = (t / t.norm()).numpy()
            print(f"{key}: {len(ids)} tokens")

    np.savez(args.out, **out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
