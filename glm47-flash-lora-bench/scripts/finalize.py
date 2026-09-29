"""Summarize a run's probe JSONL and attach it (plus config + log) to the run's wandb entry."""
import json
import statistics
import sys
from pathlib import Path

import wandb

B = Path("/workspace/data/bench")
PROJECT = "glm47-flash-lora-bench"
WARMUP = 5  # steps 1-5 excluded from timing

fw, seq, tag, rc, cfg_path = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
recs, exit_rec = [], None
probe = B / "results" / f"{tag}.probe.jsonl"
if probe.exists():
    for line in probe.read_text().splitlines():
        r = json.loads(line)
        (recs.append(r) if "step" in r and "event" not in r else None)
        if r.get("event") == "exit" and r["steps"] > 0:
            exit_rec = r

timed = [r["step_time"] for r in recs if r["step"] > WARMUP and r["step_time"]]
summary = {
    "framework": fw, "seq_len": seq, "tag": tag, "exit_code": rc,
    "steps_completed": len(recs),
    "trainable_params": recs[0].get("trainable_params") if recs else None,
    "optimizer": recs[0].get("optimizer") if recs else None,
    "tokens_per_step_measured": statistics.median(r["tokens"] for r in recs[1:]) if len(recs) > 1 else None,
    "tokens_first_step": recs[0]["tokens"] if recs else None,
    "median_step_s": statistics.median(timed) if timed else None,
    "mean_step_s": statistics.mean(timed) if timed else None,
    "stdev_step_s": statistics.stdev(timed) if len(timed) > 1 else None,
    "timed_steps": len(timed),
    "peak_reserved_gib": max((r["max_reserved_gib"] for r in recs), default=None),
    "peak_allocated_gib": max((r["max_allocated_gib"] for r in recs), default=None),
    "peak_smi_gib": max([r["smi_peak_gib"] for r in recs] + ([exit_rec["smi_peak_gib"]] if exit_rec else []), default=None),
}
if summary["median_step_s"] and summary["tokens_per_step_measured"]:
    summary["tokens_per_s"] = summary["tokens_per_step_measured"] / summary["median_step_s"]
(B / "results" / f"{tag}.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))

api = wandb.Api()
entity = api.default_entity
try:
    runs = sorted(api.runs(f"{entity}/{PROJECT}", filters={"display_name": tag}), key=lambda r: r.created_at)
except ValueError:  # project not created yet
    runs = []
run_id = runs[-1].id if runs else None
run = wandb.init(project=PROJECT, entity=entity, id=run_id, resume="allow" if run_id else None,
                 name=tag, group=fw, tags=[fw, f"seq{seq}"], reinit=True)
run.define_metric("bench/opt_step")
run.define_metric("bench/*", step_metric="bench/opt_step")
for r in recs:
    run.log({"bench/opt_step": r["step"], "bench/step_time_s": r["step_time"], "bench/tokens": r["tokens"],
             "bench/max_reserved_gib": r["max_reserved_gib"], "bench/max_allocated_gib": r["max_allocated_gib"],
             "bench/smi_peak_gib": r["smi_peak_gib"]})
for k, v in summary.items():
    run.summary[f"bench/{k}"] = v
art = wandb.Artifact(f"bench-{tag}", type="bench-run")
for p in [cfg_path, B / "logs" / f"{tag}.log", probe, B / "results" / f"{tag}.json"]:
    if Path(p).exists():
        art.add_file(str(p))
run.log_artifact(art)
run.finish(exit_code=rc)
