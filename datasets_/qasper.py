"""
datasets_/qasper.py — QASPER adapter (allenai/qasper).

Information-seeking QA over NLP papers. Each paper contains multiple questions;
this adapter explodes them into one example per question.
Context = title + abstract + full paper text (sections with headers).

allenai/qasper uses a legacy loading script, so it is loaded via the
HuggingFace-converted parquet files instead.

Answer priority per annotator: free_form_answer > extractive_spans > yes_no.
Unanswerable questions are skipped.

Config keys (under cfg["dataset"]):
    adapter:    qasper
    split:      validation | train | test   (default: validation)
    limit:      int | null                  (caps number of papers, not QA pairs)
"""

import logging
from typing import Any, Dict, Iterator, List, Optional

log = logging.getLogger(__name__)

_PARQUET = {
    "train":      "https://huggingface.co/datasets/allenai/qasper/resolve/refs%2Fconvert%2Fparquet/qasper/train/0000.parquet",
    "validation": "https://huggingface.co/datasets/allenai/qasper/resolve/refs%2Fconvert%2Fparquet/qasper/validation/0000.parquet",
    "test":       "https://huggingface.co/datasets/allenai/qasper/resolve/refs%2Fconvert%2Fparquet/qasper/test/0000.parquet",
}


class QASPERAdapter:
    def __init__(self):
        self._examples: List[Dict[str, Any]] = []

    def load(self, config: Dict[str, Any]) -> None:
        from datasets import load_dataset

        split = config.get("split", "validation")
        limit: Optional[int] = config.get("limit")

        url = _PARQUET.get(split)
        if url is None:
            raise ValueError(f"Unknown split '{split}'. Choose from: {list(_PARQUET)}")

        log.info("Loading allenai/qasper (%s) via parquet", split)
        ds = load_dataset("parquet", data_files=url, split="train")

        if limit:
            ds = ds.select(range(min(limit, len(ds))))

        self._examples = []
        for paper in ds:
            self._examples.extend(self._explode_paper(paper))

        log.info("Loaded %d QA examples from %d papers", len(self._examples), len(ds))

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self._examples)

    def __len__(self) -> int:
        return len(self._examples)

    @staticmethod
    def _build_context(paper: Dict[str, Any]) -> str:
        parts = [paper["title"], paper["abstract"]]
        ft = paper["full_text"]
        for sec_name, paras in zip(ft["section_name"], ft["paragraphs"]):
            header = f"\n## {sec_name}" if sec_name else ""
            body = "\n".join(p for p in paras if p)
            if body:
                parts.append(f"{header}\n{body}".strip())
        return "\n\n".join(p for p in parts if p)

    @staticmethod
    def _extract_gold(ann_bundle: Dict) -> List[str]:
        """ann_bundle = {'answer': [list of per-annotator answer dicts], ...}"""
        gold = []
        for a in ann_bundle.get("answer", []):
            if a.get("unanswerable"):
                continue
            if a.get("free_form_answer"):
                gold.append(a["free_form_answer"])
            elif a.get("extractive_spans"):
                gold.append(" | ".join(a["extractive_spans"]))
            elif a.get("yes_no") is not None:
                gold.append("yes" if a["yes_no"] else "no")
        return gold

    @classmethod
    def _explode_paper(cls, paper: Dict[str, Any]) -> List[Dict[str, Any]]:
        context = cls._build_context(paper)
        qas = paper["qas"]
        questions = qas["question"]
        q_ids = qas["question_id"]
        answers_per_q = qas["answers"]

        examples = []
        for q, q_id, ann_list in zip(questions, q_ids, answers_per_q):
            gold = cls._extract_gold(ann_list)
            if not gold:
                continue  # all annotators marked unanswerable
            examples.append({
                "id":       f"{paper['id']}_{q_id}",
                "question": q,
                "answer":   gold,
                "context":  context,
            })
        return examples
