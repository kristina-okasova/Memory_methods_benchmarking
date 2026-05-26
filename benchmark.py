"""
benchmark.py — Reproducible model × dataset evaluation harness.

Usage:
    python benchmark.py --config configs/memagent_hotpotqa.yaml
    python benchmark.py --config configs/memagent_hotpotqa.yaml --limit 100 --output results/run1.jsonl

Adding a new model:  create models/<name>.py implementing ModelAdapter.
Adding a new dataset: create datasets_/<name>.py implementing DatasetAdapter.
Both are auto-discovered via the registry below.
"""

import argparse
import importlib
import json
import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("benchmark")


# ---------------------------------------------------------------------------
# Abstract interfaces — implement these to plug in new models / datasets
# ---------------------------------------------------------------------------

class ModelAdapter(ABC):
    """Minimal interface every model wrapper must implement."""

    @abstractmethod
    def load(self, config: Dict[str, Any]) -> None:
        """Load/initialise the model (download weights, start server, etc.)."""

    @abstractmethod
    def predict(self, context: str, question: str) -> str:
        """Return the model's answer string for a single (context, question) pair."""

    def teardown(self) -> None:
        """Optional: clean up resources (stop server, free GPU, …)."""


class DatasetAdapter(ABC):
    """Minimal interface every dataset wrapper must implement."""

    @abstractmethod
    def load(self, config: Dict[str, Any]) -> None:
        """Download / load the dataset from the config."""

    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Yield dicts with at least: id, context, question, answer (str | list[str])."""

    @abstractmethod
    def __len__(self) -> int:
        """Return total number of examples (after any limit applied in load())."""


# ---------------------------------------------------------------------------
# Simple exact-match / F1 scorer (works for QA benchmarks like HotpotQA)
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    import re, string
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def exact_match(prediction: str, gold) -> float:
    golds = gold if isinstance(gold, list) else [gold]
    pred_norm = _normalise(prediction)
    return float(any(_normalise(g) == pred_norm for g in golds))


def token_f1(prediction: str, gold) -> float:
    from collections import Counter
    golds = gold if isinstance(gold, list) else [gold]
    best = 0.0
    pred_tokens = _normalise(prediction).split()
    for g in golds:
        gold_tokens = _normalise(g).split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------

@dataclass
class Result:
    id: str
    question: str
    gold_answer: Any
    prediction: str
    exact_match: float
    token_f1: float
    latency_s: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registry — auto-discovers adapters in models/ and datasets_/
# ---------------------------------------------------------------------------

def _load_adapter(base_package: str, name: str):
    """Import <base_package>.<name> and return the single exported adapter class.

    Discovery uses duck-typing so adapter modules don't need to import from
    benchmark.py (which would break under __main__ vs module name mismatch).
    A ModelAdapter must have load() + predict(); a DatasetAdapter must have
    load() + __iter__() + __len__(). Abstract classes are excluded.
    """
    import inspect
    module = importlib.import_module(f"{base_package}.{name}")
    for attr in dir(module):
        if attr.startswith("_"):
            continue
        obj = getattr(module, attr)
        if not isinstance(obj, type) or inspect.isabstract(obj):
            continue
        is_model   = callable(getattr(obj, "load", None)) and callable(getattr(obj, "predict", None))
        is_dataset = callable(getattr(obj, "load", None)) and callable(getattr(obj, "__iter__", None)) and callable(getattr(obj, "__len__", None))
        if is_model or is_dataset:
            return obj
    raise ImportError(f"No adapter class found in {base_package}.{name}")


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run(cfg: Dict[str, Any], limit: Optional[int], output_path: Path) -> Dict[str, float]:
    # --- Load adapters ---
    model_name   = cfg["model"]["adapter"]   # e.g. "memagent"
    dataset_name = cfg["dataset"]["adapter"] # e.g. "hotpotqa"

    log.info("Loading model adapter: %s", model_name)
    ModelClass = _load_adapter("models", model_name)
    model: ModelAdapter = ModelClass()
    model.load(cfg["model"])

    log.info("Loading dataset adapter: %s", dataset_name)
    DatasetClass = _load_adapter("datasets_", dataset_name)
    dataset: DatasetAdapter = DatasetClass()
    if limit:
        cfg["dataset"]["limit"] = limit
    dataset.load(cfg["dataset"])

    log.info("Dataset size: %d examples", len(dataset))

    # --- Inference loop ---
    results: List[Result] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as fout:
        for i, example in enumerate(dataset):
            ex_id    = example["id"]
            context  = example.get("context", "")
            question = example["question"]
            gold     = example["answer"]

            t0 = time.perf_counter()
            error = None
            prediction = ""
            try:
                prediction = model.predict(context, question)
            except Exception as exc:
                error = str(exc)
                log.warning("Example %s failed: %s", ex_id, exc)
            latency = time.perf_counter() - t0

            r = Result(
                id=ex_id,
                question=question,
                gold_answer=gold,
                prediction=prediction,
                exact_match=exact_match(prediction, gold) if not error else 0.0,
                token_f1=token_f1(prediction, gold) if not error else 0.0,
                latency_s=round(latency, 3),
                error=error,
            )
            results.append(r)
            fout.write(json.dumps(asdict(r)) + "\n")
            fout.flush()

            if (i + 1) % 10 == 0 or (i + 1) == len(dataset):
                em_so_far = sum(r.exact_match for r in results) / len(results)
                f1_so_far = sum(r.token_f1   for r in results) / len(results)
                log.info(
                    "[%d/%d]  EM=%.3f  F1=%.3f  last_latency=%.2fs",
                    i + 1, len(dataset), em_so_far, f1_so_far, latency,
                )

    model.teardown()

    # --- Aggregate metrics ---
    n = len(results)
    metrics = {
        "n_examples":   n,
        "exact_match":  round(sum(r.exact_match for r in results) / n, 4),
        "token_f1":     round(sum(r.token_f1   for r in results) / n, 4),
        "avg_latency_s":round(sum(r.latency_s  for r in results) / n, 3),
        "n_errors":     sum(1 for r in results if r.error),
    }
    metrics_path = output_path.with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"config": cfg, "metrics": metrics, "timestamp": datetime.utcnow().isoformat()}, f, indent=2)

    log.info("=== Final metrics ===")
    for k, v in metrics.items():
        log.info("  %s: %s", k, v)
    log.info("Results written to %s", output_path)
    log.info("Metrics  written to %s", metrics_path)
    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Model × dataset benchmark harness")
    parser.add_argument("--config",  required=True, help="Path to YAML config file")
    parser.add_argument("--limit",   type=int, default=None, help="Cap number of examples (for quick tests)")
    parser.add_argument("--output",  default=None, help="Output JSONL path (default: results/<model>_<dataset>_<ts>.jsonl)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    if args.output:
        output_path = Path(args.output)
    else:
        m = cfg["model"]["adapter"]
        d = cfg["dataset"]["adapter"]
        output_path = Path(f"results/{m}_{d}_{ts}.jsonl")

    run(cfg, args.limit, output_path)


if __name__ == "__main__":
    # Make sure local packages are importable when run from project root
    sys.path.insert(0, str(Path(__file__).parent))
    main()
