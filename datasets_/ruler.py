"""
datasets_/ruler.py — RULER adapter (simonjegou/ruler).

Synthetic long-context benchmark with tasks: NIAH (needle-in-a-haystack
variants), variable tracking (vt), common/frequent word extraction (cwe/fwe),
and QA (qa_1, qa_2).

Config keys (under cfg["dataset"]):
    adapter:        ruler
    context_length: 4096 | 8192 | 16384   (default: 4096)
    tasks:          list of task names, or null for all
                    e.g. [niah_single_1, niah_multikey_1, vt, qa_1]
    split:          test                   (only split available)
    limit:          int | null

Available tasks: niah_single_1/2/3, niah_multikey_1/2/3, niah_multivalue,
                 niah_multiquery, vt, cwe, fwe, qa_1, qa_2
"""

import logging
from typing import Any, Dict, Iterator, List, Optional

log = logging.getLogger(__name__)

_VALID_LENGTHS = {4096, 8192, 16384}


class RULERAdapter:
    def __init__(self):
        self._examples: List[Dict[str, Any]] = []

    def load(self, config: Dict[str, Any]) -> None:
        from datasets import load_dataset

        ctx_len: int = int(config.get("context_length", 4096))
        if ctx_len not in _VALID_LENGTHS:
            raise ValueError(f"context_length must be one of {_VALID_LENGTHS}, got {ctx_len}")

        tasks: Optional[List[str]] = config.get("tasks")
        split: str = config.get("split", "test")
        limit: Optional[int] = config.get("limit")

        log.info("Loading simonjegou/ruler (length=%d, tasks=%s)", ctx_len, tasks or "all")
        ds = load_dataset("simonjegou/ruler", str(ctx_len), split=split)

        if tasks:
            task_set = set(tasks)
            ds = ds.filter(lambda ex: ex["task"] in task_set)

        if limit:
            ds = ds.select(range(min(limit, len(ds))))

        self._examples = [self._format(ex, i) for i, ex in enumerate(ds)]
        log.info("Loaded %d examples", len(self._examples))

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self._examples)

    def __len__(self) -> int:
        return len(self._examples)

    @staticmethod
    def _format(ex: Dict[str, Any], idx: int) -> Dict[str, Any]:
        return {
            "id":       f"{ex['task']}_{idx}",
            "question": ex["question"],
            "answer":   ex["answer"],   # already a list
            "context":  ex["context"],
            "metadata": {"task": ex["task"], "max_new_tokens": ex.get("max_new_tokens", 128)},
        }
