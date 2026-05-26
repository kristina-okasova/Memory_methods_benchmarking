"""
datasets_/narrativeqa.py — NarrativeQA adapter (deepmind/narrativeqa).

QA over full books and movie scripts. Each document has multiple questions.
Context defaults to the human-written summary (~300–500 words); set
context_source: full_text to use the raw document (can be >100k tokens).

Config keys (under cfg["dataset"]):
    adapter:        narrativeqa
    split:          test | train | validation   (default: test)
    context_source: summary | full_text         (default: summary)
    limit:          int | null
"""

import hashlib
import logging
import re
from typing import Any, Dict, Iterator, List, Optional

log = logging.getLogger(__name__)


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


class NarrativeQAAdapter:
    def __init__(self):
        self._examples: List[Dict[str, Any]] = []

    def load(self, config: Dict[str, Any]) -> None:
        from datasets import load_dataset

        split = config.get("split", "test")
        context_source: str = config.get("context_source", "summary")
        limit: Optional[int] = config.get("limit")

        log.info("Loading deepmind/narrativeqa (%s, context=%s)", split, context_source)
        ds = load_dataset("deepmind/narrativeqa", split=split)

        if limit:
            ds = ds.select(range(min(limit, len(ds))))

        self._examples = [self._format(ex, context_source) for ex in ds]
        log.info("Loaded %d examples", len(self._examples))

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self._examples)

    def __len__(self) -> int:
        return len(self._examples)

    @staticmethod
    def _format(ex: Dict[str, Any], context_source: str) -> Dict[str, Any]:
        doc = ex["document"]
        question_text = ex["question"]["text"]

        if context_source == "full_text":
            context = _strip_html(doc["text"])
        else:
            # summary is a dict {"text": ..., "tokens": ...}
            context = doc["summary"]["text"].strip()

        answers = [a["text"] for a in ex["answers"]]

        # Stable id: doc id + hash of question text
        q_hash = hashlib.md5(question_text.encode()).hexdigest()[:8]
        ex_id = f"{doc['id']}_{q_hash}"

        return {
            "id":       ex_id,
            "question": question_text,
            "answer":   answers,
            "context":  context,
        }
