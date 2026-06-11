"""
models/deltamem.py — Adapter for declare-lab/delta-mem_qwen3_4b-instruct (δ-mem)

δ-mem augments a FROZEN Qwen3-4B-Instruct-2507 backbone with a small (rank-8)
online associative-memory state patched directly into its attention layers
("TSW" delta-rule adapter). Unlike Mem-alpha / ReMemR1, it is NOT served via an
OpenAI-compatible HTTP endpoint — it runs as an in-process `transformers` model
with a custom attention patch (`attach_delta_mem`) and a hand-rolled
token-by-token decode loop (`DeltaMemChatSession`).

There is no external retrieval / search step and no special prompt protocol:
the model is simply given "<context>\\n\\nQuestion: <question>\\nAnswer:" and
generates a response, exactly like the HotpotQA eval in the upstream repo
(deltamem/eval/benchmark_compare.py — HOTPOTQA_PROMPT_TEMPLATE), which uses the
same EM/F1 metrics this benchmark already computes.

Repo:           https://github.com/declare-lab/delta-Mem
Adapter weights: https://huggingface.co/declare-lab/delta-mem_qwen3_4b-instruct
Base model:     Qwen/Qwen3-4B-Instruct-2507

REQUIRES A SEPARATE VENV
-------------------------
Qwen3 needs transformers >= 4.51, which conflicts with this project's vLLM
.venv (pinned to transformers==4.46.3 for vLLM 0.8.5). Set up a dedicated venv:

    bash scripts/setup_deltamem_venv.sh

then run with:

    .deltamem_venv/bin/python benchmark.py --config configs/deltamem_hotpotqa.yaml --workers 1

(--workers 1 because this adapter loads ONE model in-process per worker — no
vLLM-style continuous batching. With a 4B model you *can* try --workers 2 if
GPU memory allows, but each worker loads its own full copy of the model.)

Config keys (all under cfg["model"]):
    adapter:             deltamem
    deltamem_repo_path:  delta-Mem                       (cloned repo, relative to project root)
    base_model_path:     Qwen/Qwen3-4B-Instruct-2507      (HF hub id or local path)
    adapter_path:        declare-lab/delta-mem_qwen3_4b-instruct  (HF hub id or local dir)
    device:              cuda:0
    dtype:               bfloat16                         (float16 | bfloat16 | float32)
    attn_implementation: sdpa                             (sdpa | eager | flash_attention_2)
    max_new_tokens:      64
    do_sample:           false
    temperature:         1.0
    top_p:               1.0
    top_k:               0
    max_context_chars:   120000     # head+tail truncation, mirrors the repo's own
                                     # MemoryAgentBench eval (clip_context_text)
    prompt_template: |               # optional override; must contain {context} and {question}
        Answer the question using only the passages below.
        Reply with a short span or yes/no only.

        {context}

        Question: {question}
        Answer:
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# Mirrors deltamem/eval/benchmark_compare.py::HOTPOTQA_PROMPT_TEMPLATE — short
# extractive-span answers (HotpotQA, MuSiQue, 2WikiMultihopQA, QASPER, RULER).
PROMPT_TEMPLATE_SHORT = (
    "Answer the question using only the passages below.\n"
    "Reply with a short span or yes/no only.\n\n"
    "{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

# Free-form variant for long-answer datasets (NarrativeQA), scored with ROUGE-L.
PROMPT_TEMPLATE_FREEFORM = (
    "Read the passage below and answer the question in your own words.\n\n"
    "{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

_TRUNCATION_MARKER = "\n\n[... context truncated ...]\n\n"


def _clip_context_chars(text: str, max_chars: int) -> str:
    """Head+tail truncation, mirroring delta-Mem's own clip_context_text()."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = _TRUNCATION_MARKER
    if max_chars <= len(marker) + 32:
        return text[-max_chars:]
    head_chars = max(1, (max_chars - len(marker)) // 3)
    tail_chars = max(1, max_chars - len(marker) - head_chars)
    return text[:head_chars].rstrip() + marker + text[-tail_chars:].lstrip()


_DTYPES = {
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float32": "float32",
}


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class DeltaMemAdapter:
    """ModelAdapter wrapping declare-lab/delta-mem (Qwen3-4B + online associative memory)."""

    def __init__(self):
        self._cfg: Dict[str, Any] = {}
        self._session = None
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # ModelAdapter interface
    # ------------------------------------------------------------------

    def load(self, config: Dict[str, Any]) -> None:
        self._cfg = config

        repo_path = self._resolve_repo_path(config.get("deltamem_repo_path", "delta-Mem"))
        repo_str = str(repo_path)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        try:
            import torch  # noqa: F401
            from deltamem.core.delta import (
                HFDeltaMemConfig,
                attach_delta_mem,
                load_delta_mem_adapter,
            )
            from deltamem.runtime.session import DeltaMemChatSession, get_dtype
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Could not import the deltamem package / its dependencies. "
                "Make sure you're running with .deltamem_venv (set up via "
                f"scripts/setup_deltamem_venv.sh) and that {repo_path} is a "
                "clone of https://github.com/declare-lab/delta-Mem"
            ) from e

        base_model_path = config.get("base_model_path", "Qwen/Qwen3-4B-Instruct-2507")
        adapter_path = config.get("adapter_path", "declare-lab/delta-mem_qwen3_4b-instruct")
        device = config.get("device", "cuda:0")
        dtype = config.get("dtype", "bfloat16")
        attn_implementation = config.get("attn_implementation", "sdpa")

        adapter_dir = self._resolve_adapter_dir(adapter_path)

        log.info(
            "Loading δ-mem: base=%s adapter=%s device=%s dtype=%s attn=%s",
            base_model_path, adapter_dir, device, dtype, attn_implementation,
        )

        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            dtype=get_dtype(dtype),
            device_map={"": device},
            attn_implementation=attn_implementation,
        ).eval()

        delta_config = HFDeltaMemConfig.from_pretrained(adapter_dir)
        attach_delta_mem(model, delta_config)
        load_delta_mem_adapter(model, adapter_dir)

        self._model = model
        self._tokenizer = tokenizer
        self._session = DeltaMemChatSession(model=model, tokenizer=tokenizer, device=device)
        log.info("δ-mem adapter ready.")

    def predict(self, context: str, question: str) -> str:
        max_context_chars = self._cfg.get("max_context_chars", 120000)
        context = _clip_context_chars(context or "", max_context_chars)

        template = self._cfg.get("prompt_template", PROMPT_TEMPLATE_SHORT)
        prompt = template.format(context=context, question=question)

        # Reset online memory state + KV-cache so examples don't leak into each other.
        self._session.reset()

        max_new_tokens = self._cfg.get("max_new_tokens", 64)
        do_sample = self._cfg.get("do_sample", False)
        temperature = self._cfg.get("temperature", 1.0)
        top_p = self._cfg.get("top_p", 1.0)
        top_k = self._cfg.get("top_k", 0)

        try:
            result = self._session.generate_reply(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        except Exception:
            # Make sure a bad example can't leave the session/model state poisoned
            # for the next call.
            self._session.reset()
            raise

        text = result.get("assistant_display") or result.get("assistant") or ""
        return text.strip()

    def teardown(self) -> None:
        try:
            import torch
            self._session = None
            self._model = None
            self._tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_repo_path(repo_path: str) -> Path:
        p = Path(repo_path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent.parent / p
        if not p.exists():
            raise FileNotFoundError(
                f"delta-Mem repo not found at {p}. "
                "Clone it with: git clone https://github.com/declare-lab/delta-Mem"
            )
        return p

    @staticmethod
    def _resolve_adapter_dir(adapter_path: str) -> str:
        """load_delta_mem_adapter() needs a local directory containing
        delta_mem_config.json + delta_mem_adapter.pt. If adapter_path isn't an
        existing local path, download a snapshot from the HF hub."""
        p = Path(adapter_path)
        if p.exists():
            return str(p)
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise ImportError("pip install huggingface_hub") from e
        log.info("Downloading δ-mem adapter weights from HF hub: %s", adapter_path)
        return snapshot_download(repo_id=adapter_path)
