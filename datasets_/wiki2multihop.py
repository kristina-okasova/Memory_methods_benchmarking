"""
datasets_/wiki2multihop.py — 2WikiMultihopQA adapter (framolfese/2WikiMultihopQA).

Multi-hop QA over Wikipedia. Context structure is identical to HotpotQA
(dict with title/sentences lists), so formatting follows the same pattern.

Config keys (under cfg["dataset"]):
    adapter:    wiki2multihop
    split:      validation | train | test   (default: validation)
    limit:      int | null
    type_filter: bridge | comparison | compositional | inference | null (default: null = all)
"""

import logging
from typing import Any, Dict, Iterator, List, Optional

log = logging.getLogger(__name__)

_DOCUMENT_PROMPT = "Document {i}:\n{document}"


class Wiki2MultihopAdapter:
    def __init__(self):
        self._examples: List[Dict[str, Any]] = []

    def load(self, config: Dict[str, Any]) -> None:
        from datasets import load_dataset

        split = config.get("split", "validation")
        limit: Optional[int] = config.get("limit")
        type_filter: Optional[str] = config.get("type_filter")

        log.info("Loading framolfese/2WikiMultihopQA (%s)", split)
        ds = load_dataset("framolfese/2WikiMultihopQA", split=split)

        if type_filter:
            ds = ds.filter(lambda ex: ex["type"] == type_filter)

        if limit:
            ds = ds.select(range(min(limit, len(ds))))

        self._examples = [self._format(ex) for ex in ds]
        log.info("Loaded %d examples", len(self._examples))

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self._examples)

    def __len__(self) -> int:
        return len(self._examples)

    @staticmethod
    def _format(ex: Dict[str, Any]) -> Dict[str, Any]:
        titles: List[str] = ex["context"]["title"]
        sentences: List[List[str]] = ex["context"]["sentences"]

        docs = [
            f"{title}\n{''.join(sents)}"
            for title, sents in zip(titles, sentences)
        ]
        context = "\n\n".join(
            _DOCUMENT_PROMPT.format(i=i + 1, document=doc)
            for i, doc in enumerate(docs)
        )
        return {
            "id":       ex["id"],
            "question": ex["question"],
            "answer":   ex["answer"],
            "context":  context,
        }
