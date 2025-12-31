#!/usr/bin/env python3
"""Generate long-run PIBERT-only configs for 2000-epoch training.

This helper keeps the original data/model hyper-parameters but:
  * filters the model list down to PIBERT-based variants so PIBERT remains the focus,
  * bumps the epoch budget (default 2000) and seed list (default 42/43/44),
  * enables EMA for a stabler tail of training,
  * writes the adjusted config under ``longrun_configs/`` so original files stay intact.

Optionally pass ``--device cuda`` (or any runner.py-compatible string) to switch the
generated configs to your Windows GPU box, ``--seeds 42`` to keep a single run, and
``--eval-every 100`` to reduce validation frequency for faster epochs.

After generating the configs you can launch long trainings via:

    python runner.py --config longrun_configs/tube_pibert_2000ep.json

Then rerun ``rebuild_history_and_export_contours.py`` once the runs finish.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent

CASE_TO_CONFIG = {
    "tube": "config_tube.json",
    "tube_p": "config_tube_p.json",
    "cylinder": "config_cylinder.json",
    "cavity": "config_cavity.json",
    "dam_p": "config_dam.json",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Emit PIBERT-only long-run configs")
    ap.add_argument(
        "--cases",
        nargs="*",
        default=sorted(CASE_TO_CONFIG.keys()),
        help="Subset of cases to process (default: all)",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=2000,
        help="Epoch budget for the long runs.",
    )
    ap.add_argument(
        "--seeds",
        default="42",
        help="Comma-separated list of seeds to cycle through (default: 42).",
    )
    ap.add_argument(
        "--outdir",
        default="longrun_configs",
        help="Directory to write adjusted config files into.",
    )
    ap.add_argument(
        "--device",
        default=None,
        help=(
            "Override the training device for the emitted configs."
            " Use values accepted by runner.py, e.g. 'cuda', 'mps', 'cpu'."
        ),
    )
    ap.add_argument(
        "--suffix",
        default="pibert_{epochs}ep",
        help="Template for the config filename suffix (format vars: case, epochs).",
    )
    ap.add_argument(
        "--eval-every",
        type=int,
        default=100,
        help="How often to run validation (in epochs).",
    )
    ap.add_argument(
        "--max-val-batches",
        type=int,
        default=0,
        help="Clamp validation loader iterations (0 = full loader).",
    )
    return ap.parse_args()


def _parse_seeds(raw: str) -> List[int]:
    seeds: List[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        seeds.append(int(chunk))
    if not seeds:
        raise ValueError("at least one seed required")
    return seeds


def _load_base(case: str) -> Dict:
    cfg_name = CASE_TO_CONFIG[case]
    with open(ROOT / cfg_name, "r") as fh:
        return json.load(fh)


def _filter_models_to_pibert(cfg: Dict) -> Dict:
    models = [m for m in cfg.get("models", []) if m.lower().startswith("pibert")]
    if not models:
        raise ValueError("Config has no PIBERT-like models to keep")
    cfg["models"] = models
    model_cfg = cfg.get("model_cfg", {})
    cfg["model_cfg"] = {k: v for k, v in model_cfg.items() if k in models}
    return cfg


def _enable_ema(cfg: Dict) -> None:
    ema = cfg.setdefault("ema", {})
    ema.setdefault("decay", 0.999)
    ema["enabled"] = True


def adjust_config(
    case: str,
    cfg: Dict,
    epochs: int,
    seeds: List[int],
    *,
    device_override: str | None,
    eval_every: int,
    max_val_batches: int,
) -> Dict:
    cfg = json.loads(json.dumps(cfg))  # deep copy via JSON
    cfg = _filter_models_to_pibert(cfg)

    train = cfg.setdefault("train", {})
    train["epochs"] = epochs
    train["seed"] = int(seeds[0])
    train["seeds"] = seeds
    train.pop("seed_repeats", None)
    train.pop("repeats", None)
    train.setdefault("warmup_epochs", max(20, epochs // 20))
    if device_override:
        train["device"] = device_override

    _enable_ema(cfg)

    eval_cfg = cfg.setdefault("eval", {})
    outdir = eval_cfg.get("outdir", f"results_{case}")
    eval_cfg["outdir"] = f"{outdir}_pibert_{epochs}ep"
    eval_cfg["eval_every"] = int(eval_every)
    eval_cfg["max_val_batches"] = int(max_val_batches)

    return cfg


def main() -> None:
    args = parse_args()
    seeds = _parse_seeds(args.seeds)
    outdir = (ROOT / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    for case in args.cases:
        if case not in CASE_TO_CONFIG:
            print(f"[skip] unknown case: {case}")
            continue
        base = _load_base(case)
        cfg = adjust_config(
            case,
            base,
            args.epochs,
            seeds,
            device_override=args.device,
            eval_every=args.eval_every,
            max_val_batches=args.max_val_batches,
        )
        suffix = args.suffix.format(case=case, epochs=args.epochs)
        fname = f"{case}_{suffix}.json"
        path = outdir / fname
        with open(path, "w") as fh:
            json.dump(cfg, fh, indent=2)
            fh.write("\n")
        print(f"[write] {path}")
        print(f"        run: python runner.py --config {path}")


if __name__ == "__main__":
    main()
