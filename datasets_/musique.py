"""
datasets_/musique.py — MuSiQue adapter (dgslibisey/MuSiQue).

Multi-hop QA (2–4 hops) over Wikipedia paragraphs.
Context is all provided paragraphs formatted as "Paragraph N:\\n<title>\\n<text>".
Only answerable examples are kept by default.

Config keys (under cfg["dataset"]):
    adapter:            musique
    split:              validation | train   (default: validation)
    limit:              int | null
    only_answerable:    true | false         (default: true)
"""

import logging
from typing import Any, Dict, Iterator, List, Optional

log = logging.getLogger(__name__)

_PARA_PROMPT = "Paragraph {i}:\n{title}\n{text}"


class MuSiQueAdapter:
    def __init__(self):
        self._examples: List[Dict[str, Any]] = []

    def load(self, config: Dict[str, Any]) -> None:
        from datasets import load_dataset

        split = config.get("split", "validation")
        limit: Optional[int] = config.get("limit")
        only_answerable: bool = config.get("only_answerable", True)

        log.info("Loading dgslibisey/MuSiQue (%s)", split)
        ds = load_dataset("dgslibisey/MuSiQue", split=split)

        if only_answerable:
            ds = ds.filter(lambda ex: ex["answerable"])

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
        paragraphs = ex["paragraphs"]
        context = "\n\n".join(
            _PARA_PROMPT.format(i=i + 1, title=p["title"], text=p["paragraph_text"])
            for i, p in enumerate(paragraphs)
        )
        # answer + answer_aliases give all accepted gold answers
        answers = [ex["answer"]] + [a for a in ex.get("answer_aliases", []) if a]
        return {
            "id":       ex["id"],
            "question": ex["question"],
            "answer":   answers,
            "context":  context,
        }
