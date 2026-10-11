#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Generate and drift-check per-crate map.md files.

Each crate may carry a map.md at its crate root. The mechanical half of the
map — source file list with each file's exported symbols, direct and reverse
dependencies — lives between explicit markers:

    <!-- crate-map:generated:begin -->
    ...
    <!-- crate-map:generated:end -->

and is regenerated from the tree by this script. Everything outside the
markers is hand-written judgement (purpose, status, gotchas) and is preserved
byte-for-byte on regeneration.

The block deliberately carries no volatile stats (line counts, item/test
counts, totals): it changes only when a file is added/removed, an exported
symbol is added/removed/renamed, or a dependency edge changes. Editing a
function body never requires a regen.

Usage:
    scripts/check-crate-maps.py <crate>...        generate/refresh those maps
    scripts/check-crate-maps.py --check [crate...]  fail on drift
        (no crate named = every map.md present under crates/)

Exit codes in --check mode (mirrors scripts/check-env-docs.py):
    0 - every checked map matches the tree
    1 - drift found (details printed, one line per finding)
    2 - usage error or a named crate has no map.md
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CRATES = ROOT / "crates"

BEGIN = "<!-- crate-map:generated:begin -->"
END = "<!-- crate-map:generated:end -->"

# Leading whitespace is REQUIRED to be optional here. Anchoring `^pub` at column
# zero silently reported 0 public items for `rdna-compute/src/gemm.rs`, a file with
# 296 of them -- every one is a method inside an `impl Gpu` block, so every one is
# indented. Any crate whose API is methods rather than free functions was
# under-counted the same way, and the drift check passed anyway because it
# regenerates the map with the same counter: a self-consistent wrong measurement.
#
# Caveat this cannot resolve by regex: a `pub fn` inside a private `mod` is not
# actually public API, so this over-counts in that case. Over-counting a few is
# strictly better than reporting 296 as 0.
PUB_ITEM = re.compile(
    r"^\s*pub\s+(?:unsafe\s+)?(?:async\s+)?(?:default\s+)?"
    r"(fn|struct|enum|trait|type|const|static|mod|use)\s+([A-Za-z_][A-Za-z0-9_]*)"
)


def public_items(path: Path) -> list[str]:
    names: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = PUB_ITEM.match(line)
        if m and m.group(2) not in names:
            names.append(m.group(2))
    return sorted(names)


def module_inventory(crate: Path) -> list[tuple[str, list[str]]]:
    """(relative path, sorted exported names) for every src/**/*.rs."""
    src = crate / "src"
    if not src.is_dir():
        return []
    return [
        (rs.relative_to(crate).as_posix(), public_items(rs))
        for rs in sorted(src.rglob("*.rs"))
    ]


def parse_manifest(crate: Path) -> dict[str, list[str]]:
    """Dependencies by kind: path / external / dev / build (crate-local path
    deps are recognised by a `path = "../` value)."""
    deps: dict[str, list[str]] = {"path": [], "external": [], "dev": [], "build": []}
    manifest = crate / "Cargo.toml"
    if not manifest.is_file():
        return deps
    section = ""
    for raw in manifest.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        m = re.match(r"\[([^\]]+)\]", line)
        if m:
            section = m.group(1).strip().strip('"')
            continue
        m = re.match(r"([A-Za-z0-9_-]+)(?:\.workspace)?\s*=\s*(.+)", line)
        if not m:
            continue
        name, value = m.group(1), m.group(2)
        base = section
        if base.startswith("target."):
            base = base.rsplit(".", 1)[-1]
        is_path = 'path = "../' in value or 'path="../' in value
        if base == "dependencies":
            deps["path" if is_path else "external"].append(name)
        elif base == "dev-dependencies":
            deps["dev"].append(name)
        elif base == "build-dependencies":
            deps["build"].append(name)
    for key in deps:
        deps[key] = sorted(set(deps[key]))
    return deps


def workspace_crates() -> list[Path]:
    return sorted(p.parent for p in CRATES.glob("*/Cargo.toml"))


def reverse_dependencies(crate: Path) -> list[str]:
    target = crate.name
    rev = []
    for other in workspace_crates():
        if other == crate:
            continue
        manifest = (other / "Cargo.toml").read_text(encoding="utf-8", errors="ignore")
        if f'path = "../{target}"' in manifest or f'path="../{target}"' in manifest:
            rev.append(other.name)
    return sorted(rev)


def dep_line(label: str, names: list[str]) -> str:
    return f"- {label}: " + (", ".join(f"`{n}`" for n in names) if names else "—")


def generated_block(crate: Path) -> str:
    modules = module_inventory(crate)
    deps = parse_manifest(crate)
    revdeps = reverse_dependencies(crate)

    lines = [
        BEGIN,
        "",
        "_Generated by `scripts/check-crate-maps.py` from the tree — do not edit "
        "inside the markers._",
        "",
        "### Modules and exported symbols",
        "",
    ]
    for rel, names in modules:
        shown = ", ".join(f"`{n}`" for n in names) if names else "—"
        lines.append(f"- [`{rel}`]({rel}): {shown}")
    lines += [
        "",
        "### Dependencies (from `Cargo.toml`)",
        "",
        dep_line("path", deps["path"]),
        dep_line("external", deps["external"]),
        dep_line("dev", deps["dev"]),
        dep_line("build", deps["build"]),
        "",
        "### Reverse dependencies",
        "",
        "- workspace crates with a path dependency on this crate: "
        + (", ".join(f"`{n}`" for n in revdeps) if revdeps else "—"),
        "",
        END,
    ]
    return "\n".join(lines)


def parse_block_modules(block: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for rel, rest in re.findall(r"^- \[`(src/[^`]+)`\]\([^)]*\): (.*)$", block, re.MULTILINE):
        out[rel] = set(re.findall(r"`([^`]+)`", rest))
    return out


def parse_block_deps(block: str) -> set[str]:
    deps: set[str] = set()
    in_deps = False
    for line in block.splitlines():
        if line.startswith("### Dependencies"):
            in_deps = True
            continue
        if line.startswith("### "):
            in_deps = False
        if in_deps and line.startswith("- ") and "—" not in line:
            deps.update(re.findall(r"`([^`]+)`", line))
    return deps


def parse_block_revdeps(block: str) -> set[str]:
    m = re.search(r"### Reverse dependencies\n\n- [^\n]*", block)
    if not m or "—" in m.group(0):
        return set()
    return set(re.findall(r"`([^`]+)`", m.group(0)))


def check_crate(crate: Path) -> list[str]:
    map_path = crate / "map.md"
    if not map_path.is_file():
        return [f"{crate.name}: no map.md"]
    text = map_path.read_text(encoding="utf-8", errors="ignore")
    if BEGIN not in text or END not in text:
        return [f"{crate.name}: map.md is missing the generated-block markers"]
    old_block = text[text.index(BEGIN):text.index(END) + len(END)]
    fresh = generated_block(crate)
    if old_block == fresh:
        return []

    problems: list[str] = []
    old_modules = parse_block_modules(old_block)
    new_modules = {rel: set(names) for rel, names in module_inventory(crate)}
    for rel in sorted(new_modules.keys() - old_modules.keys()):
        problems.append(f"{crate.name}: module exists with no map entry: {rel}")
    for rel in sorted(old_modules.keys() - new_modules.keys()):
        problems.append(f"{crate.name}: map lists a module that no longer exists: {rel}")
    for rel in sorted(new_modules.keys() & old_modules.keys()):
        for name in sorted(new_modules[rel] - old_modules[rel]):
            problems.append(f"{crate.name}: export missing from map: {rel}::{name}")
        for name in sorted(old_modules[rel] - new_modules[rel]):
            problems.append(f"{crate.name}: map lists a removed export: {rel}::{name}")

    old_deps = parse_block_deps(old_block)
    new_deps_raw = parse_manifest(crate)
    new_deps = set(new_deps_raw["path"]) | set(new_deps_raw["external"]) | set(new_deps_raw["dev"]) | set(new_deps_raw["build"])
    for name in sorted(old_deps - new_deps):
        problems.append(f"{crate.name}: stale declared dependency in map: {name}")
    for name in sorted(new_deps - old_deps):
        problems.append(f"{crate.name}: dependency missing from map: {name}")

    old_rev, new_rev = parse_block_revdeps(old_block), set(reverse_dependencies(crate))
    for name in sorted(old_rev - new_rev):
        problems.append(f"{crate.name}: stale reverse dependency in map: {name}")
    for name in sorted(new_rev - old_rev):
        problems.append(f"{crate.name}: reverse dependency missing from map: {name}")

    if not problems:
        problems.append(
            f"{crate.name}: generated block is stale — "
            f"run scripts/check-crate-maps.py {crate.name}"
        )
    return problems


def generate_crate(crate: Path) -> bool:
    map_path = crate / "map.md"
    block = generated_block(crate)
    if map_path.is_file():
        text = map_path.read_text(encoding="utf-8", errors="ignore")
        if BEGIN not in text or END not in text:
            print(f"{crate.name}: map.md exists but has no generated-block markers", file=sys.stderr)
            return False
        new_text = text[:text.index(BEGIN)] + block + text[text.index(END) + len(END):]
        if new_text != text:
            map_path.write_text(new_text, encoding="utf-8")
            print(f"{crate.name}: refreshed generated block")
        else:
            print(f"{crate.name}: already up to date")
        return True

    skeleton = f"""# {crate.name} — map

> **Status:** `production` / `research` / `legacy` — pick exactly one
> (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** see the layering table in
> [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md) — do not restate it here.

## Purpose

<!-- hand-written: what this crate owns and why it exists. Prefer pointing at
     the `//!` docs in `src/lib.rs` over copying prose. -->

## Gotchas

<!-- hand-written: what not to do here; traps a reader will hit. -->

## Crate map

{block}
"""
    map_path.write_text(skeleton, encoding="utf-8")
    print(f"{crate.name}: wrote new map.md skeleton with generated block")
    return True


def main(argv: list[str]) -> int:
    check = "--check" in argv
    names = [a for a in argv if not a.startswith("--")]

    if check:
        if names:
            crates = []
            for name in names:
                crate = CRATES / name
                if not (crate / "map.md").is_file():
                    print(f"{name}: no map.md", file=sys.stderr)
                    return 2
                crates.append(crate)
        else:
            crates = sorted(p.parent for p in CRATES.glob("*/map.md"))
            if not crates:
                print("crate-maps: no map.md files present; nothing to check")
                return 0
        problems: list[str] = []
        for crate in crates:
            problems.extend(check_crate(crate))
        without = len(list(CRATES.glob("*/Cargo.toml"))) - len(crates)
        if problems:
            print("crate maps have drifted from the tree:")
            for p in problems:
                print(f"  {p}")
            return 1
        print(
            f"crate-maps: {len(crates)} map(s) match the tree "
            f"({without} crates without a map, not enforced)"
        )
        return 0

    if not names:
        print(__doc__, file=sys.stderr)
        return 2
    ok = True
    for name in names:
        crate = CRATES / name
        if not crate.is_dir():
            print(f"{name}: no such crate under crates/", file=sys.stderr)
            ok = False
            continue
        ok = generate_crate(crate) and ok
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
