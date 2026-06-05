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
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Queue
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

def run(cfg: Dict[str, Any], limit: Optional[int], output_path: Path,
        workers: int = 1) -> Dict[str, float]:
    """Run a model against a dataset.

    Parameters
    ----------
    workers:
        Number of parallel example workers.  Each worker gets its own adapter
        instance so there are no shared-state races.  The vLLM / SGLang server
        handles concurrent requests via continuous batching, so the GPU stays
        busy across all workers.  Recommended: 4–16 for server-mode adapters.
    """
    # --- Load adapters ---
    model_name   = cfg["model"]["adapter"]   # e.g. "memagent"
    dataset_name = cfg["dataset"]["adapter"] # e.g. "hotpotqa"

    log.info("Loading dataset adapter: %s", dataset_name)
    DatasetClass = _load_adapter("datasets_", dataset_name)
    dataset: DatasetAdapter = DatasetClass()
    if limit:
        cfg["dataset"]["limit"] = limit
    dataset.load(cfg["dataset"])
    examples = list(dataset)   # materialise so workers can index freely
    total = len(examples)
    log.info("Dataset size: %d examples", total)

    # --- Resume: load already-completed results from an existing output file ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_map: Dict[int, Result] = {}   # index → result

    # Build id → index map for fast lookup
    id_to_idx = {ex["id"]: i for i, ex in enumerate(examples)}

    if output_path.exists():
        prior_lines = 0
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    ex_id = d["id"]
                    if ex_id in id_to_idx:
                        idx = id_to_idx[ex_id]
                        results_map[idx] = Result(**{
                            k: d[k] for k in Result.__dataclass_fields__
                        })
                        prior_lines += 1
                except Exception:
                    pass   # corrupt line — will be re-run
        if prior_lines:
            log.info("Resume: found %d already-completed results in %s — skipping them.",
                     prior_lines, output_path)

    pending = [(i, ex) for i, ex in enumerate(examples) if i not in results_map]
    log.info("%d examples remaining to process.", len(pending))

    # --- Build a pool of adapter instances (one per worker) ---
    actual_workers = min(workers, max(len(pending), 1))
    log.info("Loading %d × model adapter: %s", actual_workers, model_name)
    ModelClass = _load_adapter("models", model_name)
    adapter_pool: Queue = Queue()
    for _ in range(actual_workers):
        m: ModelAdapter = ModelClass()
        m.load(cfg["model"])
        adapter_pool.put(m)

    # --- Thread-safe helpers ---
    write_lock   = threading.Lock()
    counter_lock = threading.Lock()
    completed    = [len(results_map)]   # start counter at already-done count

    def process_one(idx: int, example: Dict[str, Any]) -> Result:
        ex_id    = example["id"]
        context  = example.get("context", "")
        question = example["question"]
        gold     = example["answer"]

        adapter = adapter_pool.get()
        t0 = time.perf_counter()
        error = None
        prediction = ""
        try:
            prediction = adapter.predict(context, question)
        except Exception as exc:
            error = str(exc)
            log.warning("Example %s failed: %s", ex_id, exc)
        finally:
            adapter_pool.put(adapter)
        latency = time.perf_counter() - t0

        return Result(
            id=ex_id,
            question=question,
            gold_answer=gold,
            prediction=prediction,
            exact_match=exact_match(prediction, gold) if not error else 0.0,
            token_f1=token_f1(prediction, gold) if not error else 0.0,
            latency_s=round(latency, 3),
            error=error,
        )

    # --- Parallel inference (append to existing file) ---
    with open(output_path, "a") as fout, \
         ThreadPoolExecutor(max_workers=actual_workers) as executor:

        future_to_idx = {
            executor.submit(process_one, i, ex): i
            for i, ex in pending
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            r = future.result()   # re-raises if process_one threw unexpectedly

            with write_lock:
                results_map[idx] = r
                fout.write(json.dumps(asdict(r)) + "\n")
                fout.flush()

            with counter_lock:
                completed[0] += 1
                done = completed[0]

            if done % 10 == 0 or done == total:
                # Compute running metrics over whatever has finished so far
                done_results = list(results_map.values())
                em_so_far = sum(x.exact_match for x in done_results) / len(done_results)
                f1_so_far = sum(x.token_f1   for x in done_results) / len(done_results)
                log.info(
                    "[%d/%d]  EM=%.3f  F1=%.3f  last_latency=%.2fs",
                    done, total, em_so_far, f1_so_far, r.latency_s,
                )

    # Teardown all adapter instances
    while not adapter_pool.empty():
        adapter_pool.get_nowait().teardown()

    # --- Aggregate metrics (over all results in original order) ---
    results = [results_map[i] for i in range(total)]
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
    parser.add_argument("--output",  default=None,
                        help="Output JSONL path (default: results/<model>_<dataset>_<ts>.jsonl). "
                             "Pass the same path as a previous run to resume it.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel example workers (default: 1). "
                             "Each worker gets its own adapter instance; the inference "
                             "server handles concurrent requests. Try 8–16 for server-mode models.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the most recent output file for this model+dataset. "
                             "Ignored if --output is explicitly specified.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    m = cfg["model"]["adapter"]
    d = cfg["dataset"]["adapter"]
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    if args.output:
        output_path = Path(args.output)
    elif args.resume:
        # Find the most recent matching output file in results/
        candidates = sorted(Path("results").glob(f"{m}_{d}_*.jsonl"), reverse=True)
        if candidates:
            output_path = candidates[0]
            log.info("Resuming from %s", output_path)
        else:
            log.warning("--resume specified but no previous file found for %s_%s; starting fresh.", m, d)
            output_path = Path(f"results/{m}_{d}_{ts}.jsonl")
    else:
        output_path = Path(f"results/{m}_{d}_{ts}.jsonl")

    run(cfg, args.limit, output_path, workers=args.workers)


if __name__ == "__main__":
    # Make sure local packages are importable when run from project root
    sys.path.insert(0, str(Path(__file__).parent))
    main()
