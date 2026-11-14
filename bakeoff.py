"""
Step 2 (optional but recommended): train every candidate in config.yaml briefly and rank them.

    python bakeoff.py                 # short runs (3 head + 6 full epochs each)
    python bakeoff.py --full          # full-length runs

Prints and saves runs/bakeoff.md. Pick the winner by FIELD accuracy and macro-F1 first, then size/latency.
Desktop CPU latency is only a rough guide: the final call should use the app's Benchmark screen on a real phone.
"""
import argparse
from pathlib import Path

from common import load_cfg
from train import run

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml")
ap.add_argument("--full", action="store_true")
ap.add_argument("--only", nargs="*", help="subset of model names")
a = ap.parse_args()
cfg = load_cfg(a.config)

results = []
for name in (a.only or cfg["models"]):
    print(f"\n################ {name} ################")
    try:
        # default = quick screening run (about 5-10 min per model on a mid-range GPU); --full = the config epochs
        kw = {} if a.full else dict(epochs_head=1, epochs_full=3, samples_per_epoch=8000, val_subset=2000)
        results.append(run(cfg, name, tag=f"bake_{name}", **kw))
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"FAILED {name}: {e}")

results.sort(key=lambda m: (m["test"]["acc_field"] or 0) + m["test"]["macro_f1"], reverse=True)
lines = ["| model | params (M) | test acc | field acc | macro-F1 | top-3 | ECE (calibrated) | CPU ms |",
         "|---|---|---|---|---|---|---|---|"]
for m in results:
    t = m["test"]
    fa = f"{t['acc_field']:.4f}" if t["acc_field"] is not None else "n/a"
    lines.append(f"| {m['model']} | {m['params_m']:.1f} | {t['acc']:.4f} | {fa} | {t['macro_f1']:.4f} | "
                 f"{t['top3']:.4f} | {t['ece']:.4f} | {m['cpu_ms']:.0f} |")
table = "\n".join(lines)
print("\n" + table)
Path(cfg["paths"]["runs_dir"]).mkdir(exist_ok=True)
(Path(cfg["paths"]["runs_dir"]) / "bakeoff.md").write_text(table + "\n", encoding="utf-8")
