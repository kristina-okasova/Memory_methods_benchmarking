"""
run_suite.py — Run a model against multiple (or all) datasets.

Discovers every adapter in datasets_/ automatically. Each dataset is run
sequentially and a summary table is printed at the end.

Usage:
    # run all datasets with a given model config
    python run_suite.py --model-config configs/memagent_hotpotqa.yaml

    # run specific datasets only
    python run_suite.py --model-config configs/memagent_hotpotqa.yaml \\
        --datasets hotpotqa musique wiki2multihop

    # limit examples per dataset (useful for quick smoke tests)
    python run_suite.py --model-config configs/memagent_hotpotqa.yaml --limit 50

    # override output directory
    python run_suite.py --model-config configs/memagent_hotpotqa.yaml --output-dir results/suite_run1
"""

import argparse
import importlib
import inspect
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from benchmark import run, DatasetAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_suite")


def _discover_datasets(datasets_dir: Path) -> List[str]:
    """Return sorted list of dataset adapter names found in datasets_/."""
    names = []
    for f in sorted(datasets_dir.glob("*.py")):
        if f.name.startswith("_"):
            continue
        mod_name = f"datasets_.{f.stem}"
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:
            log.warning("Could not import %s: %s", mod_name, e)
            continue
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if (
                isinstance(obj, type)
                and not inspect.isabstract(obj)
                and callable(getattr(obj, "load", None))
                and callable(getattr(obj, "__iter__", None))
            ):
                names.append(f.stem)
                break
    return names


def main():
    parser = argparse.ArgumentParser(description="Run a model against multiple datasets")
    parser.add_argument(
        "--model-config", required=True,
        help="Path to any existing per-dataset YAML — only the 'model' section is used",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Dataset adapter names to run (default: all discovered in datasets_/)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap examples per dataset (for quick tests)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for result files (default: results/suite_<timestamp>/)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel example workers per dataset (default: 1). "
             "Try 8–16 for server-mode models (memalpha, rememr1).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from the most recent results directory for this model. "
             "Already-completed examples are skipped automatically.",
    )
    args = parser.parse_args()

    with open(args.model_config) as f:
        base_cfg = yaml.safe_load(f)

    model_cfg = base_cfg["model"]
    model_name = model_cfg["adapter"]

    datasets_dir = Path(__file__).parent / "datasets_"
    available = _discover_datasets(datasets_dir)
    log.info("Discovered dataset adapters: %s", available)

    targets = args.datasets if args.datasets else available
    unknown = [d for d in targets if d not in available]
    if unknown:
        log.error("Unknown dataset(s): %s. Available: %s", unknown, available)
        sys.exit(1)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.resume:
        # Find the most recent suite_* directory that contains a result file for this model
        results_root = Path("results")
        candidates = sorted(results_root.glob("suite_*"), reverse=True)
        output_dir = None
        for d in candidates:
            if any(d.glob(f"{model_name}_*.jsonl")):
                output_dir = d
                break
        if output_dir is None:
            log.error(
                "--resume specified but no previous results found for model '%s' "
                "under results/suite_*/. Starting a fresh run.", model_name
            )
            output_dir = Path(f"results/suite_{ts}")
        else:
            log.info("Resuming from %s", output_dir)
    else:
        output_dir = Path(f"results/suite_{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: Dict[str, dict] = {}

    for dataset_name in targets:
        log.info("=" * 60)
        log.info("Running  model=%s  dataset=%s", model_name, dataset_name)
        log.info("=" * 60)

        # Load per-dataset config if it exists, else fall back to defaults
        dataset_cfg_path = Path(f"configs/{model_name}_{dataset_name}.yaml")
        if dataset_cfg_path.exists():
            with open(dataset_cfg_path) as f:
                cfg = yaml.safe_load(f)
            cfg["model"] = model_cfg  # model section always comes from --model-config
        else:
            cfg = {"model": model_cfg, "dataset": {"adapter": dataset_name}}

        if args.limit:
            cfg["dataset"]["limit"] = args.limit

        output_path = output_dir / f"{model_name}_{dataset_name}.jsonl"
        try:
            metrics = run(cfg, limit=None, output_path=output_path, workers=args.workers)
            all_metrics[dataset_name] = metrics
        except Exception as exc:
            log.error("Dataset %s failed: %s", dataset_name, exc, exc_info=True)
            all_metrics[dataset_name] = {"error": str(exc)}

    # Summary table
    log.info("")
    log.info("=" * 60)
    log.info("SUITE SUMMARY  (model=%s)", model_name)
    log.info("=" * 60)
    header = f"{'dataset':<22}  {'n':>6}  {'EM':>7}  {'F1':>7}  {'errors':>7}"
    log.info(header)
    log.info("-" * len(header))
    for ds, m in all_metrics.items():
        if "error" in m:
            log.info("%-22s  %s", ds, m["error"])
        else:
            log.info(
                "%-22s  %6d  %7.3f  %7.3f  %7d",
                ds, m.get("n_examples", 0), m.get("exact_match", 0),
                m.get("token_f1", 0), m.get("n_errors", 0),
            )

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"model": model_name, "timestamp": ts, "metrics": all_metrics}, f, indent=2)
    log.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
