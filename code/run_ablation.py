"""Hyperparameter ablation sweeps over train_evaluate.py configs, mirroring
the GDHF project's run_ablation.py pattern: expand a grid spec (dot-path
parameter -> list of values) into one fully-resolved config per combination,
write them out, and run each through train_evaluate.py -- same CV harness,
WandB logging, and results_summary.csv each config already uses, just swept
instead of hand-written one at a time.

Usage:
    python run_ablation.py --grid configs/ablation_svm_kernel.yaml
    python run_ablation.py --grid configs/ablation_mlp_mil.yaml --dry-run
"""
from __future__ import annotations

import argparse
import itertools
import logging
import subprocess
import sys
from pathlib import Path

import yaml

from train_evaluate import _deep_merge, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_ablation")


def _set_path(override: dict, key: str, value) -> None:
    """Sets override[key] (dot-separated) = value, UNLESS value is itself a
    dict of {dot.path: val}, in which case each of those paths is set
    instead -- lets one grid dimension set several params together, e.g.
    pairing an optimizer choice with the learning rate it actually needs
    (Adam and SGD don't share a sensible default lr)."""
    if isinstance(value, dict):
        for sub_key, sub_val in value.items():
            _set_path(override, sub_key, sub_val)
        return
    parts = key.split(".")
    d = override
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def expand_grid(grid: dict) -> list[tuple[dict, dict]]:
    """grid: {"train.dropout": [0.1, 0.2, 0.3], ...} (dot-separated param path
    -> list of values, or list of {dot.path: val} dicts for a compound
    dimension). Returns one (override, raw_values) pair per combination --
    raw_values keeps the un-expanded per-key value for slug generation."""
    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys]
    combos = []
    for values in itertools.product(*value_lists):
        override: dict = {}
        raw_values = dict(zip(keys, values))
        for key, value in raw_values.items():
            _set_path(override, key, value)
        combos.append((override, raw_values))
    return combos


def combo_slug(grid_keys: list[str], raw_values: dict) -> str:
    parts = []
    for key in grid_keys:
        value = raw_values[key]
        if isinstance(value, dict):
            parts.append("_".join(f"{k.split('.')[-1]}{v}" for k, v in value.items()))
        else:
            parts.append(f"{key.split('.')[-1]}{value}")
    return "_".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("configs/ablation_generated"))
    parser.add_argument("--dry-run", action="store_true", help="only generate configs, don't run them")
    parser.add_argument("--entrypoint", default="train_evaluate.py")
    args = parser.parse_args()

    with open(args.grid) as f:
        spec = yaml.safe_load(f)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    grid_keys = list(spec["grid"].keys())
    combos = expand_grid(spec["grid"])
    n_total = len(spec["base_configs"]) * len(combos)
    log.info(f"{len(spec['base_configs'])} base config(s) x {len(combos)} grid combo(s) = {n_total} runs")

    generated: list[Path] = []
    for base_name in spec["base_configs"]:
        base_cfg = load_config(Path("configs") / base_name)
        for override, raw_values in combos:
            cfg = _deep_merge(base_cfg, override)
            cfg["run_name"] = f"{base_cfg['run_name']}_{combo_slug(grid_keys, raw_values)}"
            out_path = args.out_dir / f"{cfg['run_name']}.yaml"
            with open(out_path, "w") as f:
                yaml.safe_dump(cfg, f)
            generated.append(out_path)

    log.info(f"Generated {len(generated)} configs in {args.out_dir}")
    if args.dry_run:
        return

    for i, cfg_path in enumerate(generated):
        log.info(f"[{i + 1}/{len(generated)}] running {cfg_path.name}")
        subprocess.run([sys.executable, args.entrypoint, "--config", str(cfg_path)], check=True)


if __name__ == "__main__":
    main()
