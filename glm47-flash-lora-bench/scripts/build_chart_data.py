"""Collect measured results into one JSON blob for the chart page."""
import json
from pathlib import Path

B = Path("/workspace/data/bench")
R = B / "results"


def res(tag):
    p = R / f"{tag}.json"
    return json.loads(p.read_text()) if p.exists() else None


def trace(tag):
    p = R / f"{tag}.probe.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    return [round(r["step_time"], 3) for r in rows if "event" not in r and r.get("step_time")]


FW = ["axolotl", "unsloth", "llamafactory", "primerl"]
# Axolotl rows use its own default checkpointing (reentrant); the first pass forced use_reentrant=False
TAG = {("axolotl", k): f"axolotl-opt-{k}" for k in ("4k", "8k", "16k")}
# Prime-RL: best measured setting per length (torch.compile off was faster at 4k/8k, on at 16k)
TAG.update({("primerl", "4k"): "primerl-nocompile-4k", ("primerl", "8k"): "primerl-nocompile-8k"})
main = {}
for fw in FW:
    for k in ("4k", "8k", "16k"):
        r = res(TAG.get((fw, k), f"{fw}-{k}"))
        r2 = res(f"{fw}-{k}-r2") if fw != "axolotl" else None
        main.setdefault(fw, {})[k] = None if not r or not r["median_step_s"] else {
            "s": round(r["mean_step_s"], 3), "tps": round(32768 / r["mean_step_s"]), "gib": round(r["peak_reserved_gib"], 1),
            "sd": round(r["stdev_step_s"], 3), "repeat_s": r2 and r2["mean_step_s"] and round(r2["mean_step_s"], 3)}
mem = {}
for fw in FW:
    for k in ("32k", "64k"):
        tag = {("axolotl", "32k"): "mem-axolotl-opt-32k", ("axolotl", "64k"): "mem-axolotl-opt-64k"}.get((fw, k), f"mem-{fw}-{k}")
        r = res(tag)
        mem.setdefault(fw, {})[k] = {"ok": bool(r and r["median_step_s"]), "offload": (fw, k) in {("axolotl", "64k"), ("unsloth", "64k")} or (fw == "unsloth"),
                                     "s": r and r["mean_step_s"] and round(r["mean_step_s"], 2),
                                     "gib": r and r["peak_reserved_gib"] and round(r["peak_reserved_gib"], 1)}
research = {k: (lambda r: r and {"s": r["mean_step_s"] and round(r["mean_step_s"], 3),
                                "sd": r["stdev_step_s"] and round(r["stdev_step_s"], 3),
                                "gib": r["peak_reserved_gib"] and round(r["peak_reserved_gib"], 1),
                                "rc": r["exit_code"], "steps": r["steps_completed"]})(res(k))
            for k in ["ab-axolotl-base-4k", "ab-axolotl-gcoff-4k", "ab-axolotl-nocce-4k",
                      "gcfreeze-axolotl-4k", "gcfreeze-axolotl-8k", "gcfreeze-axolotl-16k",
                      "cceexact-axolotl-4k", "cceexact-axolotl-16k", "ccefp32-axolotl-4k", "kern-mlp-axolotl-4k"]}
traces = {k: trace(k) for k in ["axolotl-4k", "ab-axolotl-base-4k", "ab-axolotl-gcoff-4k", "gcfreeze-axolotl-4k", "llamafactory-4k"]}
gc = {}
for fw in FW:
    p = R / f"gclog-{fw}-4k.probe.jsonl"
    if not p.exists():
        continue
    ev = [json.loads(l) for l in p.read_text().splitlines()]
    steps = [e for e in ev if "event" not in e]
    if len(steps) < 2:
        continue
    t0, t1 = steps[0]["t"], steps[-1]["t"]
    w = [e for e in ev if e.get("event") == "gc" and t0 <= e["t"] <= t1]
    st = next((e for e in ev if e.get("event") == "gc_stats"), {})
    gc[fw] = {"steps": len(steps) - 1, "tracked": st.get("tracked_objects"),
              **{f"gen{g}_n": sum(1 for e in w if e["gen"] == g) for g in (0, 1, 2)},
              **{f"gen{g}_s": round(sum(e["dur"] for e in w if e["gen"] == g), 3) for g in (0, 1, 2)}}
# measured in scripts/cce_model_check.py (one 1k-token micro-batch, real model, LoRA grad norm)
cce = {"random": {"hf": 1.82607, "cce_default": 35.03145, "cce_accum_e_fp32": 15.66837, "cce_exact": 1.81759},
       "real": {"hf": 0.85097, "cce_default": 0.92762, "cce_accum_e_fp32": 0.86315, "cce_exact": 0.85059}}
scan = json.loads((B / "research" / "lora_kernel_scan.json").read_text())
from collections import Counter
kern = {"counts": Counter(v["status"] for v in scan.values()),
        "assert": sorted(k for k, v in scan.items() if v["status"] == "ASSERT")}
main_default = {}
for k in ("4k", "8k", "16k"):
    r = res(f"axolotl-reentrant-{k}")
    main_default[k] = {"s": round(r["mean_step_s"], 3), "tps": round(32768 / r["mean_step_s"]), "gib": round(r["peak_reserved_gib"], 1)}
out = {"main_default": main_default, "main": main, "mem": mem, "research": research, "traces": traces, "gc": gc, "cce": cce, "kernels": kern}
(B / "research" / "chart_data.json").write_text(json.dumps(out))
print(json.dumps({k: v for k, v in out.items() if k in ("main", "gc", "research")}, indent=0)[:2500])
