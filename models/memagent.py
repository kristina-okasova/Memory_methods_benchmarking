"""
models/memagent.py — Adapter for BytedTsinghua-SIA/MemAgent
============================================================

MemAgent serves its model via a vLLM-compatible OpenAI endpoint.
This adapter supports two modes controlled by config:

  mode: "local"   — starts a local vLLM server (requires GPU)
  mode: "remote"  — connects to an already-running endpoint

Config keys (all under cfg["model"]):
    adapter:          memagent          # selects this file
    mode:             local | remote
    model_name_or_path: BytedTsinghua-SIA/RL-MemoryAgent-14B
    base_url:         http://localhost:8000/v1   # remote mode
    api_key:          token-abc                  # remote mode (or set MEMAGENT_API_KEY)
    tensor_parallel_size: 2                      # local mode
    gpu_memory_utilization: 0.90                 # local mode
    max_model_len:    32768
    segment_len:      4096   # MemAgent chunking window
    memory_len:       1024   # MemAgent memory panel length
    max_new_tokens:   256
    temperature:      0.0
"""

import logging
import os
import subprocess
import time
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


# MemAgent system / instruction prompts (matches the paper's agent workflow)
_SYSTEM_PROMPT = (
    "You are a helpful assistant that processes long documents using a memory-augmented "
    "reading strategy. You will receive document chunks one at a time along with a "
    "running memory panel. Update the memory panel after each chunk, then answer the "
    "final question using the accumulated memory."
)

_CHUNK_TEMPLATE = (
    "### Memory (from previous chunks)\n{memory}\n\n"
    "### Current Chunk\n{chunk}\n\n"
    "### Instruction\n"
    "Read the current chunk carefully. "
    "Update the memory panel to preserve all information that may be needed to answer "
    "the question: {question}\n"
    "Return ONLY the updated memory panel as plain text."
)

_FINAL_TEMPLATE = (
    "### Memory (accumulated)\n{memory}\n\n"
    "### Question\n{question}\n\n"
    "### Instruction\n"
    "Based solely on the memory panel above, give a concise factual answer. "
    "Return ONLY the answer, no explanation."
)


class MemAgentAdapter:
    """Wraps MemAgent (served by vLLM) as a drop-in ModelAdapter."""

    def __init__(self):
        self._client = None
        self._server_proc: Optional[subprocess.Popen] = None
        self._cfg: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # ModelAdapter interface
    # ------------------------------------------------------------------

    def load(self, config: Dict[str, Any]) -> None:
        self._cfg = config
        mode = config.get("mode", "remote")

        if mode == "local":
            self._start_local_server(config)
        else:
            log.info("Connecting to remote MemAgent endpoint at %s", config.get("base_url"))

        self._client = self._build_client(config)
        log.info("MemAgent adapter ready (mode=%s)", mode)

    def predict(self, context: str, question: str) -> str:
        """Chunk-based MemAgent inference loop."""
        segment_len = self._cfg.get("segment_len", 4096)
        chunks = self._chunk_text(context, segment_len)

        if not chunks:
            # No context — ask directly
            return self._call_llm([
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": f"Question: {question}\nAnswer:"},
            ])

        # Multi-turn memory update
        memory = ""
        for chunk in chunks:
            prompt = _CHUNK_TEMPLATE.format(memory=memory or "(empty)", chunk=chunk, question=question)
            memory = self._call_llm([
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ])

        # Final answer from accumulated memory
        final_prompt = _FINAL_TEMPLATE.format(memory=memory, question=question)
        answer = self._call_llm([
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": final_prompt},
        ])
        return answer.strip()

    def teardown(self) -> None:
        if self._server_proc is not None:
            log.info("Stopping local vLLM server (pid=%d)…", self._server_proc.pid)
            self._server_proc.terminate()
            self._server_proc = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_text(text: str, chunk_size: int) -> list[str]:
        """Split on whitespace boundaries into chunks of ~chunk_size chars."""
        words = text.split()
        chunks, current, current_len = [], [], 0
        for word in words:
            current.append(word)
            current_len += len(word) + 1
            if current_len >= chunk_size:
                chunks.append(" ".join(current))
                current, current_len = [], 0
        if current:
            chunks.append(" ".join(current))
        return chunks

    def _call_llm(self, messages: list) -> str:
        max_tokens   = self._cfg.get("max_new_tokens", 256)
        temperature  = self._cfg.get("temperature", 0.0)
        model        = self._cfg.get("model_name_or_path", "BytedTsinghua-SIA/RL-MemoryAgent-14B")

        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _build_client(config: Dict[str, Any]):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("pip install openai>=1.0") from e

        base_url = config.get("base_url", "http://localhost:8000/v1")
        api_key  = config.get("api_key") or os.environ.get("MEMAGENT_API_KEY", "token-placeholder")
        return OpenAI(base_url=base_url, api_key=api_key)

    def _start_local_server(self, config: Dict[str, Any]) -> None:
        model   = config.get("model_name_or_path", "BytedTsinghua-SIA/RL-MemoryAgent-14B")
        tp      = config.get("tensor_parallel_size", 1)
        gpu_mem = config.get("gpu_memory_utilization", 0.90)
        max_len = config.get("max_model_len", 32768)
        port    = 8000

        cmd = [
            "vllm", "serve", model,
            "--tensor-parallel-size", str(tp),
            "--gpu-memory-utilization", str(gpu_mem),
            "--max-model-len", str(max_len),
            "--port", str(port),
        ]
        log.info("Starting vLLM server: %s", " ".join(cmd))
        self._server_proc = subprocess.Popen(cmd)

        # Poll until the server is up
        import urllib.request, urllib.error
        deadline = time.time() + 300
        while time.time() < deadline:
            try:
                urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
                log.info("vLLM server is up.")
                return
            except Exception:
                time.sleep(5)
        raise RuntimeError("vLLM server did not start within 5 minutes")
