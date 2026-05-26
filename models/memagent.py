"""
models/memagent.py — Adapter for BytedTsinghua-SIA/MemAgent

Matches MemAgent's training-time inference setup from recurrent/interface.py:
  - Token-based context chunking (chunk_size = 5000 tokens by default)
  - Center-crop truncation for contexts longer than max_context_len tokens
  - Memory update and final answer templates from the paper
  - Answer extracted from \\boxed{} format (matches training reward)

Config keys (all under cfg["model"]):
    adapter:              memagent
    mode:                 local | remote
    model_name_or_path:   BytedTsinghua-SIA/RL-MemoryAgent-7B
    base_url:             http://localhost:8000/v1   # for remote / local
    api_key:              123-abc
    tensor_parallel_size: 1           # local mode only
    gpu_memory_utilization: 0.90      # local mode only
    max_model_len:        8192        # local mode only
    chunk_size:           5000        # tokens per chunk
    max_context_len:      120000      # max context tokens; center-crop if longer
    max_new_tokens:       1024
    temperature:          0.0         # greedy for eval (training used 0.7)
    top_p:                1.0
"""

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from benchmark import ModelAdapter  # noqa: E402

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exact templates from MemAgent/recurrent/interface.py
# ---------------------------------------------------------------------------

_TEMPLATE = (
    "You are presented with a problem, a section of an article that may contain "
    "the answer to the problem, and a previous memory. Please read the provided "
    "section carefully and update the memory with the new information that helps "
    "to answer the problem. Be sure to retain all relevant details from the "
    "previous memory while adding any new, useful information.\n\n"
    "<problem> \n{prompt}\n</problem>\n\n"
    "<memory>\n{memory}\n</memory>\n\n"
    "<section>\n{chunk}\n</section>\n\n"
    "Updated memory:\n"
)

_TEMPLATE_FINAL = (
    "You are presented with a problem and a previous memory. Please answer the "
    "problem based on the previous memory and put the answer in \\boxed{{}}.\n\n"
    "<problem> \n{prompt}\n</problem>\n\n"
    "<memory>\n{memory}\n</memory>\n\n"
    "Your answer:\n"
)

_NO_MEMORY = "No previous memory"


# ---------------------------------------------------------------------------
# Boxed-answer extraction (mirrors taskutils/memory_data/hotpotqa_verifier.py)
# ---------------------------------------------------------------------------

def _extract_boxed(text: str) -> Optional[str]:
    """Pull the innermost \\boxed{...} content from model output."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    if "\\boxed " in text[idx:]:
        after = text[idx + len("\\boxed "):]
        return after.split("$")[0].strip()
    i = idx + len("\\boxed")
    if i >= len(text) or text[i] != "{":
        return None
    depth, start = 0, i
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i]
        i += 1
    return None


class MemAgentAdapter(ModelAdapter):
    """ModelAdapter wrapping MemAgent served by vLLM via OpenAI-compatible API."""

    def __init__(self):
        self._client = None
        self._tokenizer = None
        self._server_proc: Optional[subprocess.Popen] = None
        self._cfg: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # ModelAdapter interface
    # ------------------------------------------------------------------

    def load(self, config: Dict[str, Any]) -> None:
        self._cfg = config
        mode = config.get("mode", "remote")

        log.info("Loading tokenizer from %s", config.get("model_name_or_path"))
        self._tokenizer = self._load_tokenizer(config["model_name_or_path"])

        if mode == "local":
            self._start_local_server(config)
        else:
            log.info("Connecting to remote endpoint at %s", config.get("base_url"))

        self._client = self._build_client(config)
        log.info("MemAgent adapter ready (mode=%s)", mode)

    def predict(self, context: str, question: str) -> str:
        """MemAgent recurrent memory-update loop, then boxed-answer extraction."""
        max_ctx = self._cfg.get("max_context_len", 120000)
        chunk_size = self._cfg.get("chunk_size", 5000)

        # Encode context; center-crop if too long (matches training data.truncation='center')
        input_ids = self._tokenizer.encode(context)
        if len(input_ids) > max_ctx:
            half = max_ctx // 2
            input_ids = input_ids[:half] + input_ids[-half:]

        memory = _NO_MEMORY
        for i in range(0, max(1, len(input_ids)), chunk_size):
            chunk_ids = input_ids[i : i + chunk_size]
            chunk_text = self._tokenizer.decode(chunk_ids, skip_special_tokens=True)
            msg = _TEMPLATE.format(prompt=question, memory=memory, chunk=chunk_text)
            memory = self._call_llm(msg)
            if not memory:
                return ""

        final_msg = _TEMPLATE_FINAL.format(prompt=question, memory=memory)
        raw = self._call_llm(final_msg)
        if not raw:
            return ""

        extracted = _extract_boxed(raw)
        return (extracted or raw).strip()

    def teardown(self) -> None:
        if self._server_proc is not None:
            log.info("Stopping local vLLM server (pid=%d)…", self._server_proc.pid)
            self._server_proc.terminate()
            self._server_proc = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_llm(self, user_content: str) -> str:
        """Single API call; uses user-only message to match MemAgent's training setup."""
        max_tokens = self._cfg.get("max_new_tokens", 1024)
        temperature = self._cfg.get("temperature", 0.0)
        top_p = self._cfg.get("top_p", 1.0)
        model = self._cfg.get("model_name_or_path", "BytedTsinghua-SIA/RL-MemoryAgent-7B")

        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_content}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _load_tokenizer(model_name_or_path: str):
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError("pip install transformers") from e
        return AutoTokenizer.from_pretrained(model_name_or_path)

    @staticmethod
    def _build_client(config: Dict[str, Any]):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("pip install openai>=1.0") from e
        base_url = config.get("base_url", "http://localhost:8000/v1")
        api_key = config.get("api_key") or os.environ.get("MEMAGENT_API_KEY", "123-abc")
        return OpenAI(base_url=base_url, api_key=api_key)

    def _start_local_server(self, config: Dict[str, Any]) -> None:
        model = config.get("model_name_or_path", "BytedTsinghua-SIA/RL-MemoryAgent-7B")
        tp = config.get("tensor_parallel_size", 1)
        gpu_mem = config.get("gpu_memory_utilization", 0.90)
        max_len = config.get("max_model_len", 8192)
        port = 8000

        cmd = [
            "vllm", "serve", model,
            "--tensor-parallel-size", str(tp),
            "--gpu-memory-utilization", str(gpu_mem),
            "--max-model-len", str(max_len),
            "--port", str(port),
        ]
        log.info("Starting vLLM server: %s", " ".join(cmd))
        self._server_proc = subprocess.Popen(cmd)

        import urllib.request
        deadline = time.time() + 300
        while time.time() < deadline:
            try:
                urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
                log.info("vLLM server is up.")
                return
            except Exception:
                time.sleep(5)
        raise RuntimeError("vLLM server did not start within 5 minutes")
