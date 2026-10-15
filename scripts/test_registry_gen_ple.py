#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""Flash-Next PLE row slots (`ple_q4` / `ple_q8`).

A split base publishes its PLE rows as `<file>.ple-q4` / `<file>.ple-q8`. The
curated overlay names the slots; `registry_gen.py` pins each against the live
Hugging Face tree (LFS oid + size), exactly like the other sidecar kinds.
Entries without PLE fields come out unchanged.

No network: `repo_tree` is stubbed.
"""
import copy
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "registry_gen", Path(__file__).parent / "registry_gen.py"
)
rg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rg)

REPO = "test/split-flash"
BASE = "qwen3.8-flash-next-base.mq3-xts"
Q4 = f"{BASE}.ple-q4"
Q8 = f"{BASE}.ple-q8"
BASE_BYTES = 60_000_000_000
Q4_BYTES = 28_800_000_000
Q8_BYTES = 54_400_000_000


def _tree() -> dict:
    return {
        BASE: {"lfs": {"oid": "base-oid", "size": BASE_BYTES}, "size": BASE_BYTES},
        Q4: {"lfs": {"oid": "q4-oid", "size": Q4_BYTES}, "size": Q4_BYTES},
        Q8: {"lfs": {"oid": "q8-oid", "size": Q8_BYTES}, "size": Q8_BYTES},
    }


def _curated() -> dict:
    entry = {
        "repo": REPO,
        "file": BASE,
        "size_gb": 60.0,
        "min_vram_gb": 8,
        "desc": "test split base",
        "ple_q4": {"file": Q4},
        "ple_q8": {"file": Q8},
    }
    return {"models": {"qwen3.8:flash-next-mq3": entry}, "aliases": {}}


def _build(monkeypatch, curated: dict, tree: dict | None = None):
    monkeypatch.setattr(rg, "repo_tree", lambda repo, token: tree if tree is not None else _tree())
    return rg.build_registry(curated, None)


def test_ple_slots_are_pinned_from_the_live_tree(monkeypatch):
    registry, errors = _build(monkeypatch, _curated())
    assert errors == [], errors
    entry = registry["models"]["qwen3.8:flash-next-mq3"]
    assert entry["ple_q4"] == {"file": Q4, "sha256": "q4-oid", "size_bytes": Q4_BYTES}
    assert entry["ple_q8"] == {"file": Q8, "sha256": "q8-oid", "size_bytes": Q8_BYTES}
    assert entry["sha256"] == "base-oid"
    assert entry["quant"] == "mq3"
    assert entry["arch_id"] == 16


def test_one_slot_entry_carries_only_that_slot(monkeypatch):
    curated = _curated()
    del curated["models"]["qwen3.8:flash-next-mq3"]["ple_q8"]
    registry, errors = _build(monkeypatch, curated)
    assert errors == [], errors
    entry = registry["models"]["qwen3.8:flash-next-mq3"]
    assert "ple_q8" not in entry
    assert entry["ple_q4"]["sha256"] == "q4-oid"


def test_entries_without_ple_fields_are_untouched(monkeypatch):
    curated = _curated()
    for key in ("ple_q4", "ple_q8"):
        del curated["models"]["qwen3.8:flash-next-mq3"][key]
    registry, errors = _build(monkeypatch, curated)
    assert errors == [], errors
    entry = registry["models"]["qwen3.8:flash-next-mq3"]
    assert not any(key.startswith("ple") for key in entry), entry


def test_missing_slot_file_fails_closed(monkeypatch):
    tree = _tree()
    del tree[Q4]
    registry, errors = _build(monkeypatch, _curated(), tree)
    assert registry is None
    assert any("ple_q4" in e and "not found" in e for e in errors), errors


def test_non_lfs_slot_has_no_digest_pin_and_fails_closed(monkeypatch):
    tree = _tree()
    tree[Q8] = {"size": Q8_BYTES}
    registry, errors = _build(monkeypatch, _curated(), tree)
    assert registry is None
    assert any("ple_q8" in e and "sha256" in e for e in errors), errors


def test_build_does_not_mutate_the_curated_overlay(monkeypatch):
    curated = _curated()
    before = copy.deepcopy(curated)
    _build(monkeypatch, curated)
    assert curated == before
