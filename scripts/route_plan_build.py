#!/usr/bin/env -S python3 -I
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Build a checked-in route plan from route walks (route_plan_walk.py).

Every (module, symbol) the walked dispatch bound is attributed through its module
handle to the HIPFIRE_MODULE_LOAD_AUDIT record that loaded it (HIP: source SHA-256
and recipe; embedded: image SHA-256). The `ar` rows are the union of the AR walks,
the `native_mtp` rows the union of every walk (the MTP route falls back to AR).
`kernel_registry::route_entries` reads the result; its tests check that every row
resolves to the registry recipe or embedded image with the recorded identity.

usage: route_plan_build.py --out PLAN.tsv --family NAME --model M
                           --walk ar=DIR [--walk ar=DIR ...] --walk native_mtp=DIR [...]
"""
import argparse
import hashlib
import json
import pathlib
import re
import struct

# hipfire-runtime loader_api.rs `kernel_dtype_for_quant_type`, by DType Debug name.
DTYPES = {1: "F16", 2: "F32", 3: "Q8_0", 6: "HFQ4G256", 13: "MQ4G256", 15: "MQ6G256", 16: "BF16",
          44: "MQ4G256V2", 47: "MQ6G256V2", 48: "MQ5G256V2", 49: "MQ3G256V2", 50: "MQ2G256V2",
          53: "MQ4G128V2", 55: "FP8E4M3B128", 56: "FP8E4M3PT"}
# The `qwen4 state:` QSA description (hipfire-loader carriers.rs) to the axis value
# `resolve_qwen4_route_plan` derives from `QsaKvFormat`.
QSA_NAMES = {"fp8": "fp8", "fp8 (gfx942 FNUZ codes, F32 index keys)": "fp8-fnuz",
             "q8 (Q8_0 blocks of 32)": "q8", "bf16 (exact F32 state)": "f32"}


def header_tensors(model):
    """(name, dtype, shape) of every HFQM header tensor; reads no payload."""
    with open(model, "rb") as f:
        magic, version, arch, count, meta, data = struct.unpack("<4sIIIQQ", f.read(32))
        assert magic == b"HFQM" and version == 1 and 32 <= meta <= data <= 64 << 20, model
        f.seek(0)
        header = f.read(data)
    depth, quoted, escape, end = 0, False, False, None
    for i, b in enumerate(header[meta:], meta):
        if quoted:
            if escape:
                escape = False
            elif b == 92:
                escape = True
            elif b == 34:
                quoted = False
        elif b == 34:
            quoted = True
        elif b == 123:
            depth += 1
        elif b == 125:
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    pos = end
    (n,) = struct.unpack_from("<I", header, pos)
    pos += 4
    assert n == count
    out = []
    for _ in range(n):
        (length,) = struct.unpack_from("<H", header, pos)
        pos += 2
        name = header[pos:pos + length].decode()
        pos += length
        quant, ndim = struct.unpack_from("<BB", header, pos)
        pos += 2
        shape = struct.unpack_from("<" + "I" * ndim, header, pos)
        pos += 4 * ndim + 12
        out.append((name, DTYPES.get(quant, "Raw"), shape))
    return arch, out


def layout_sha256(tensors):
    """`kernel_registry::route_layout_sha256`: sorted `name\\tdtype\\tshape` lines."""
    lines = sorted(f"{name}\t{dtype}\t{','.join(str(d) for d in shape)}\n" for name, dtype, shape in tensors)
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def summarize(walk):
    """(hip, embedded, direct, policy) observed by one walk."""
    result = json.loads((walk / "result.json").read_text())
    if not result["ok"]:
        raise SystemExit(f"{walk}: walk not ok (rc={result['rc']}, errors={result['errors']})")
    pending = None
    handles = {}
    hip, emb, direct = {}, {}, []
    # Policy axes, named as the loader's `resolve_qwen4_route_plan` names them.
    policy = {"qsa_f16_arena": "off", "qsa_pooled_bf16": "off"}
    arch = None
    for line in (walk / "stderr.log").read_text(errors="replace").splitlines():
        state = re.match(r"  qwen4 state: QSA (.+) K/V \((\S+) context storage\), GDN (\S+) recurrent$", line)
        if state:
            policy["qsa"] = QSA_NAMES[state.group(1)]
            policy["kv_backend"], policy["gdn"] = state.group(2), state.group(3)
        if line.startswith("  qwen4 QSA F16 arena: on"):
            policy["qsa_f16_arena"] = "on"
        if line.startswith("  qwen4 QSA pooled BF16 mirror: on"):
            policy["qsa_pooled_bf16"] = "on"
        if not line.startswith("HIPFIRE_MODULE_LOAD_AUDIT\t"):
            continue
        parts = line.split("\t")
        kind = parts[1]
        if kind in ("hip", "embedded"):
            if pending is not None:
                raise SystemExit(f"{walk}: audit record without a module load: {pending}")
            rec = json.loads(parts[2])
            arch = arch or rec["arch"]
            pending = (kind, rec)
        elif kind in ("load", "load_data"):
            handle = parts[2]
            if pending is None:
                direct.append(line)
                handles[handle] = None
                continue
            handles[handle] = pending
            pending = None
        elif kind == "get_function":
            handle, symbol = parts[2], parts[3]
            if handle not in handles:
                raise SystemExit(f"{walk}: get_function on an unknown handle: {line}")
            entry = handles[handle]
            if entry is None:
                continue
            k, rec = entry
            if k == "hip":
                ident = (rec["source_sha256"], tuple(rec["flags"]), rec["scheduler_profile"])
                prev = hip.setdefault(rec["module"], {"ident": ident, "symbols": set()})
                if prev["ident"] != ident:
                    raise SystemExit(f"{walk}: two recipes for {rec['module']}")
                prev["symbols"].add(symbol)
            else:
                prev = emb.setdefault(rec["module"], {"sha": rec["image_sha256"], "symbols": set()})
                if prev["sha"] != rec["image_sha256"]:
                    raise SystemExit(f"{walk}: two images for {rec['module']}")
                prev["symbols"].add(symbol)
    if pending is not None:
        raise SystemExit(f"{walk}: trailing audit record without a module load")
    command = json.loads((walk / "command.json").read_text())
    return arch, hip, emb, direct, policy, command


def merge(into, walk_hip, walk_emb):
    hip, emb = into
    for module, v in walk_hip.items():
        prev = hip.setdefault(module, {"ident": v["ident"], "symbols": set()})
        if prev["ident"] != v["ident"]:
            raise SystemExit(f"walks disagree on the recipe of {module}")
        prev["symbols"] |= v["symbols"]
    for module, v in walk_emb.items():
        prev = emb.setdefault(module, {"sha": v["sha"], "symbols": set()})
        if prev["sha"] != v["sha"]:
            raise SystemExit(f"walks disagree on the image of {module}")
        prev["symbols"] |= v["symbols"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--model", required=True, help="the walked model (layout fingerprint)")
    ap.add_argument("--walk", action="append", required=True, help="ROUTE=DIR, ROUTE in ar|native_mtp")
    a = ap.parse_args()

    model_arch, tensors = header_tensors(a.model)
    routes = {"ar": ({}, {}), "native_mtp": ({}, {})}
    arch, max_seq, axes, receipts, direct_total = None, set(), {}, [], 0
    for spec in a.walk:
        route, path = spec.split("=", 1)
        walk = pathlib.Path(path)
        w_arch, hip, emb, direct, policy, command = summarize(walk)
        if arch not in (None, w_arch):
            raise SystemExit("walks span two architectures")
        arch = w_arch
        merge(routes["native_mtp"], hip, emb)
        if route == "ar":
            merge(routes["ar"], hip, emb)
        elif route != "native_mtp":
            raise SystemExit(f"unknown route {route}")
        max_seq.add(command["load"]["params"]["max_seq"])
        if "qsa" not in policy:
            raise SystemExit(f"{walk}: no qwen4 state line in the load log")
        for name, value in policy.items():
            axes.setdefault(name, set()).add(value)
        direct_total += len(direct)
        env = {k: v for k, v in command["env"].items()
               if k.startswith("HIPFIRE_") and k not in ("HIPFIRE_HOME", "HIPFIRE_LOCK_DIR", "HIPFIRE_KERNEL_CACHE",
                                                        "HIPFIRE_DEVICES", "HIPFIRE_MODULE_LOAD_AUDIT")}
        receipts.append(f"# walk {route} {walk.name} daemon_sha256={command['daemon_sha256'][:16]} "
                        f"model={command['load']['model']} requests={command['requests']} env={json.dumps(env, sort_keys=True)}"
                        f" direct_loads={len(direct)}")
    if len(max_seq) != 1:
        raise SystemExit(f"walks disagree on max_seq: {max_seq}")
    lines = [
        "# Generated by scripts/route_plan_build.py from HIPFIRE_MODULE_LOAD_AUDIT route walks",
        "# (scripts/route_plan_walk.py); do not edit. Columns: route kind module symbols sha256",
        "# (HIP: effective source; embedded: image). native_mtp rows include every ar row.",
        f"#! family {a.family}",
        f"#! model_arch {model_arch}",
        f"#! arch {arch}",
        f"#! layout {layout_sha256(tensors)}",
        f"#! max_seq {max_seq.pop()}",
        *(f"#! axis {name} {' '.join(sorted(values))}" for name, values in sorted(axes.items())),
        *receipts,
    ]
    counts = {}
    for route in ("ar", "native_mtp"):
        hip, emb = routes[route]
        for module, v in sorted(hip.items()):
            lines.append(f"{route}\thip\t{module}\t{','.join(sorted(v['symbols']))}\t{v['ident'][0]}")
        for module, v in sorted(emb.items()):
            lines.append(f"{route}\tembedded\t{module}\t{','.join(sorted(v['symbols']))}\t{v['sha']}")
        counts[route] = {"hip": len(hip), "embedded": len(emb),
                         "symbols": sum(len(v["symbols"]) for v in [*hip.values(), *emb.values()])}
    pathlib.Path(a.out).write_text("\n".join(lines) + "\n")
    print(json.dumps({"arch": arch, "routes": counts, "direct_loads": direct_total}))


if __name__ == "__main__":
    main()
