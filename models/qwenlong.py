"""
models/qwenlong.py — Adapter for Tongyi-Zhiwen/QwenLong-L1-32B

QwenLong-L1 is a 32B-parameter long-context REASONING model (RL-trained from a
Qwen2.5-32B-Instruct backbone), served via vLLM's OpenAI-compatible API. Unlike
memagent / rememr1 / memalpha, it has NO chunking, recurrent memory, or external
retrieval — the entire context is given to the model in a single prompt (default
context window 32,768 tokens; up to 131,072 with YaRN rope scaling). It serves as
a "brute-force long-context" BASELINE for comparison against the memory-augmented
adapters in this benchmark.

Replicates the eval setup from Qwen-Doc/QwenLong-L1/eval/*.py
(https://github.com/Tongyi-Zhiwen/Qwen-Doc/tree/main/QwenLong-L1/eval):
  - template_0shot prompt ($DOC$ / $Q$ placeholders, "Therefore, the answer is ...")
  - <think>...</think> stripping (extract_solution)
  - "...the answer is X." suffix extraction (extract_answer)

Repo:    https://github.com/Tongyi-Zhiwen/Qwen-Doc/tree/main/QwenLong-L1
Weights: Tongyi-Zhiwen/QwenLong-L1-32B      (bf16,  ~64GB — tight on a single H100)
         Tongyi-Zhiwen/QwenLong-L1-32B-AWQ  (int4,  ~18GB — recommended)

Requires an OpenAI-compatible server (vLLM >= 0.7.3):
    vllm serve Tongyi-Zhiwen/QwenLong-L1-32B-AWQ \\
        --quantization awq --max-model-len 32768 --port 8000

  (most datasets in this benchmark — HotpotQA distractor, MuSiQue, 2WikiMultihopQA,
  QASPER, NarrativeQA-summary — fit comfortably under 32K tokens, so the default
  131K YaRN context is usually unnecessary. To enable it, pass --max-model-len
  131072 and --rope-scaling '{"rope_type":"yarn","factor":4.0,
  "original_max_position_embeddings":32768}' to vllm serve, or set
  rope_scaling / max_model_len in local mode below.)

Config keys (all under cfg["model"]):
    adapter:              qwenlong
    mode:                 local | remote        (default: remote)
    model_name_or_path:   Tongyi-Zhiwen/QwenLong-L1-32B-AWQ
    base_url:             http://localhost:8000/v1
    api_key:              123-abc
    # ---- local mode only (ignored when mode: remote) ----
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.90
    max_model_len:        32768
    quantization:         awq      # set to null/omit for the bf16 checkpoint
    rope_scaling:                  # optional dict, passed through to vllm serve
    #   rope_type: yarn
    #   factor: 4.0
    #   original_max_position_embeddings: 32768
    # ---- inference parameters (match official eval: temperature 0.7, top_p 0.95) ----
    max_context_len:      30000    # max INPUT tokens; center-crop if longer
    max_new_tokens:       4096     # generation budget — the model "thinks" before
                                    # answering, so this is the main cost/runtime knob
    temperature:          0.7
    top_p:                0.95
"""

import json
import logging
import os
import subprocess
import time
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template — verbatim from QwenLong-L1/eval/*.py (template_0shot)
# ---------------------------------------------------------------------------

_TEMPLATE = (
    'Please read the following text and answer the question below.\n\n'
    '<text>\n$DOC$\n</text>\n\n'
    '$Q$\n\n'
    'Format your response as follows: "Therefore, the answer is (insert answer here)".'
)


# ---------------------------------------------------------------------------
# Output post-processing — mirrors extract_solution() / extract_answer()
# ---------------------------------------------------------------------------

def _strip_think(text: str) -> str:
    """Return everything after the final </think> tag, or the full text if absent."""
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


def _extract_answer(response: str) -> str:
    """Pull the answer out of "...the answer is X." (case-sensitive substring,
    matching the official eval's `"the answer is" in response` check)."""
    response = response.replace("*", "")
    if "the answer is" in response:
        ans = response.rsplit("the answer is", 1)[-1].strip()
        ans = ans.replace("<｜Assistant｜>", "").replace("<｜end▁of▁sentence｜>", "").strip()
        ans = ans.strip(".").strip()
        ans = ans.strip('"').strip("'").strip()
        if ans.startswith("(") and ans.endswith(")"):
            ans = ans[1:-1].strip()
        return ans
    return response.strip()


class QwenLongAdapter:
    """ModelAdapter wrapping QwenLong-L1-32B served by vLLM via OpenAI-compatible API."""

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
        log.info("QwenLong-L1 adapter ready (mode=%s)", mode)

    def predict(self, context: str, question: str) -> str:
        """Single-shot: full (truncated) context + question -> answer.

        No chunking or memory loop — QwenLong-L1's RL training relies on its
        extended context window to read the whole document at once.
        """
        max_ctx = self._cfg.get("max_context_len", 30000)

        # Center-crop the context if it exceeds the configured token budget
        # (same strategy as memagent/rememr1's truncation='center').
        input_ids = self._tokenizer.encode(context)
        if len(input_ids) > max_ctx:
            half = max_ctx // 2
            input_ids = input_ids[:half] + input_ids[-half:]
            context = self._tokenizer.decode(input_ids, skip_special_tokens=True)

        prompt = _TEMPLATE.replace("$DOC$", context.strip()).replace("$Q$", question.strip())

        raw = self._call_llm(prompt)
        if not raw:
            return ""

        after_think = _strip_think(raw)
        return _extract_answer(after_think)

    def teardown(self) -> None:
        if self._server_proc is not None:
            log.info("Stopping local vLLM server (pid=%d)…", self._server_proc.pid)
            self._server_proc.terminate()
            self._server_proc = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_llm(self, user_content: str) -> str:
        max_tokens = self._cfg.get("max_new_tokens", 4096)
        temperature = self._cfg.get("temperature", 0.7)
        top_p = self._cfg.get("top_p", 0.95)
        model = self._cfg.get("model_name_or_path", "Tongyi-Zhiwen/QwenLong-L1-32B-AWQ")

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
        api_key = config.get("api_key") or os.environ.get("QWENLONG_API_KEY", "123-abc")
        return OpenAI(base_url=base_url, api_key=api_key)

    def _start_local_server(self, config: Dict[str, Any]) -> None:
        model = config.get("model_name_or_path", "Tongyi-Zhiwen/QwenLong-L1-32B-AWQ")
        tp = config.get("tensor_parallel_size", 1)
        gpu_mem = config.get("gpu_memory_utilization", 0.90)
        max_len = config.get("max_model_len", 32768)
        quantization = config.get("quantization", "awq")
        rope_scaling = config.get("rope_scaling")
        port = 8000

        cmd = [
            "vllm", "serve", model,
            "--tensor-parallel-size", str(tp),
            "--gpu-memory-utilization", str(gpu_mem),
            "--max-model-len", str(max_len),
            "--port", str(port),
        ]
        if quantization:
            cmd += ["--quantization", str(quantization)]
        if rope_scaling:
            cmd += ["--rope-scaling", json.dumps(rope_scaling)]

        log.info("Starting vLLM server: %s", " ".join(cmd))
        self._server_proc = subprocess.Popen(cmd)

        import urllib.request
        deadline = time.time() + 600  # 32B model — allow extra time to load
        while time.time() < deadline:
            try:
                urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
                log.info("vLLM server is up.")
                return
            except Exception:
                time.sleep(5)
        raise RuntimeError("vLLM server did not start within 10 minutes")
