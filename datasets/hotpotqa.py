"""
datasets_/hotpotqa.py — HotpotQA adapter for the benchmark harness.

Loads hotpotqa/hotpot_qa from HuggingFace and formats each example the same
way MemAgent does in taskutils/memory_data/dataset_process.py:

  context = "Document 1:\\n<title>\\n<sentences joined>\\n\\nDocument 2:\\n..."

The 'distractor' config (10 docs per example) is used by default — this is
what MemAgent was trained and evaluated on.

Config keys (under cfg["dataset"]):
    adapter:    hotpotqa
    hf_config:  distractor | fullwiki   (default: distractor)
    split:      validation | train       (default: validation)
    limit:      int | null               (cap examples for quick tests)
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from benchmark import DatasetAdapter  # noqa: E402

log = logging.getLogger(__name__)

_DOCUMENT_PROMPT = "Document {i}:\n{document}"


class HotpotQAAdapter(DatasetAdapter):
    """DatasetAdapter for hotpotqa/hotpot_qa (HuggingFace)."""

    def __init__(self):
        self._examples: List[Dict[str, Any]] = []

    def load(self, config: Dict[str, Any]) -> None:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError("pip install datasets") from e

        hf_config = config.get("hf_config", "distractor")
        split = config.get("split", "validation")
        limit: Optional[int] = config.get("limit", None)

        log.info("Loading hotpotqa/hotpot_qa (%s / %s)", hf_config, split)
        ds = load_dataset("hotpotqa/hotpot_qa", hf_config, split=split)
        if limit:
            ds = ds.select(range(min(limit, len(ds))))

        self._examples = [self._format(ex) for ex in ds]
        log.info("Loaded %d examples", len(self._examples))

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self._examples)

    def __len__(self) -> int:
        return len(self._examples)

    # ------------------------------------------------------------------

    @staticmethod
    def _format(ex: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a raw HotpotQA example into the harness dict format."""
        titles: List[str] = ex["context"]["title"]
        sentences: List[List[str]] = ex["context"]["sentences"]

        # Build per-document text: "Title\nSentence1Sentence2..."
        docs = [
            f"{title}\n{''.join(sents)}"
            for title, sents in zip(titles, sentences)
        ]
        # Format like MemAgent's dataset_process.py
        context = "\n\n".join(
            _DOCUMENT_PROMPT.format(i=i + 1, document=doc)
            for i, doc in enumerate(docs)
        )

        return {
            "id":       ex["id"],
            "question": ex["question"],
            "answer":   ex["answer"],   # single string in HotpotQA
            "context":  context,
        }
