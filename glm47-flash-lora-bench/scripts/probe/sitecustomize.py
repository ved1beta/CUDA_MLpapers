"""Framework-agnostic benchmark probe, loaded via PYTHONPATH when BENCH_OUT is set.

Hooks every torch optimizer step (global post-hook) and the vocab embedding forward, so the
same measurement code runs under Axolotl, Unsloth, LLaMA-Factory and Prime-RL.
Writes one JSON line per optimizer step to $BENCH_OUT.
"""
import os

if os.environ.get("BENCH_OUT") and os.environ.get("LOCAL_RANK", "0") == "0":
    import atexit
    import sys
    import json
    import threading
    import time

    import torch
    from torch.optim.optimizer import register_optimizer_step_post_hook

    import gc

    if os.environ.get("BENCH_GC_DISABLE") == "1":
        gc.disable()

    _OUT = os.environ["BENCH_OUT"]

    if os.environ.get("BENCH_CCE_KW"):
        # inject CCE options (e.g. impl=cce_exact) that axolotl's plugin does not expose
        import builtins

        _cce_kw = json.loads(os.environ["BENCH_CCE_KW"])
        _real_import = builtins.__import__

        def _import(name, *a, **k):
            mod = _real_import(name, *a, **k)
            if name == "cut_cross_entropy.transformers.patch" or (
                name.startswith("cut_cross_entropy") and "cut_cross_entropy.transformers.patch" in sys.modules
            ):
                P = sys.modules.get("cut_cross_entropy.transformers.patch")
                if P is not None and hasattr(P, "cce_patch") and not getattr(P.cce_patch, "_bench", False):
                    orig = P.cce_patch

                    def cce_patch(*args, **kwargs):
                        kwargs.update(_cce_kw)
                        _write({"event": "cce_patch", "kwargs": {k: str(v) for k, v in kwargs.items()}})
                        return orig(*args, **kwargs)

                    cce_patch._bench = True
                    P.cce_patch = cce_patch
            return mod

        builtins.__import__ = _import

    _gc_state = {"t": None, "frozen": False}
    if os.environ.get("BENCH_GC_LOG") == "1":
        def _gc_cb(phase, info):
            if phase == "start":
                _gc_state["t"] = time.perf_counter()
            elif _gc_state["t"] is not None:
                _write({"event": "gc", "gen": info["generation"], "dur": time.perf_counter() - _gc_state["t"],
                        "collected": info["collected"], "step": _state["step"], "t": time.time() - _state["t0"]})

        gc.callbacks.append(_gc_cb)
    _VOCAB = int(os.environ.get("BENCH_VOCAB", "154880"))
    _state = {"step": 0, "t_last": None, "tokens": 0, "smi_peak_mib": 0, "t0": time.time()}

    def _smi_sampler():
        try:
            import pynvml

            pynvml.nvmlInit()
            idx = int((os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0])
            h = pynvml.nvmlDeviceGetHandleByIndex(idx)
            while True:
                used = pynvml.nvmlDeviceGetMemoryInfo(h).used // 2**20
                _state["smi_peak_mib"] = max(_state["smi_peak_mib"], used)
                time.sleep(0.2)
        except Exception as e:  # noqa: BLE001
            _state["smi_error"] = repr(e)

    threading.Thread(target=_smi_sampler, daemon=True).start()

    _orig_embed_forward = torch.nn.Embedding.forward

    def _embed_forward(self, input):  # noqa: A002
        if self.num_embeddings >= _VOCAB and torch.is_grad_enabled():
            _state["tokens"] += input.numel()
            if not _gc_state["frozen"]:
                _gc_state["frozen"] = True
                try:
                    import collections

                    def _tname(o):
                        t = type(o)
                        m = getattr(t, "__module__", None)
                        m = m if isinstance(m, str) else None
                        return f"{(m or '?').split('.')[0]}.{getattr(t, '__qualname__', t.__name__)}"

                    objs = gc.get_objects()
                    top = collections.Counter(_tname(o) for o in objs)
                    _write({"event": "gc_stats", "tracked_objects": len(objs), "top_types": top.most_common(25),
                            "counts": gc.get_count(), "thresholds": gc.get_threshold()})
                    del objs, top
                except Exception as e:  # noqa: BLE001  never let the probe break training
                    _write({"event": "gc_stats_error", "err": repr(e)})
                if os.environ.get("BENCH_GC_FREEZE") == "1":
                    gc.collect()
                    gc.freeze()
                    _write({"event": "gc_freeze", "frozen": gc.get_freeze_count()})
        return _orig_embed_forward(self, input)

    torch.nn.Embedding.forward = _embed_forward

    def _write(rec):
        with open(_OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _post_step(optimizer, args, kwargs):
        torch.cuda.synchronize()
        now = time.time()
        _state["step"] += 1
        rec = {
            "step": _state["step"],
            "t": now - _state["t0"],
            "step_time": None if _state["t_last"] is None else now - _state["t_last"],
            "tokens": _state["tokens"],
            "max_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "smi_peak_gib": _state["smi_peak_mib"] / 1024,
        }
        if _state["step"] == 1:
            params = [p for g in optimizer.param_groups for p in g["params"] if p.requires_grad]
            rec["trainable_params"] = sum(p.numel() for p in params)
            rec["optimizer"] = type(optimizer).__name__
        _write(rec)
        _prof_step = int(os.environ.get("BENCH_PROFILE_STEP", "0"))
        if _prof_step:
            import cProfile
            import pstats

            if _state["step"] == _prof_step - 1:
                _state["prof"] = cProfile.Profile()
                _state["prof"].enable()
            elif _state["step"] == _prof_step and "prof" in _state:
                _state["prof"].disable()
                with open(_OUT.replace(".probe.jsonl", ".profile.txt"), "w") as f:
                    ps = pstats.Stats(_state["prof"], stream=f)
                    ps.sort_stats("ncalls").print_stats(60)
                    ps.sort_stats("tottime").print_stats(40)
                ps.dump_stats(_OUT.replace(".probe.jsonl", ".prof"))
        _state["t_last"] = now
        _state["tokens"] = 0

    register_optimizer_step_post_hook(_post_step)

    @atexit.register
    def _final():
        _write({"event": "exit", "steps": _state["step"], "smi_peak_gib": _state["smi_peak_mib"] / 1024,
                "smi_error": _state.get("smi_error")})
