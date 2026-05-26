"""
models/memalpha.py — Adapter for YuWangX/Memalpha-4B (Mem-alpha)

Mem-alpha uses a two-phase approach:
  1. Memory building — context is chunked and fed to the agent (status='memorie'),
     which calls the model to extract episodic/semantic memories from each chunk.
  2. QA — accumulated memories + question are answered via:
       direct mode  : agent.chat(question, status='chat') using local model + search tools
       server mode  : POST to memory_server.py /batch_process endpoint

The chunk prompt template matches Mem-alpha's training setup (unified_prompt from
config/prompts_wrt_datasource.yaml).

Modes
-----
mode: direct   No memory_server.py needed. The same local model handles both
               memory building (memorie) and QA (chat). Simpler to run.

mode: server   Memory building runs locally. QA is handled by memory_server.py,
               which wraps a separate vLLM endpoint.
               Requires two services to be running before benchmarking:
                 1. vLLM:  vllm serve YuWangX/Memalpha-4B --port 8001
                 2. Flask: cd Mem-alpha && python memory_server.py \\
                               --server_url http://localhost:8001/v1 \\
                               --model_name YuWangX/Memalpha-4B \\
                               --port 5005

Config keys (all under cfg["model"]):
    adapter:            memalpha
    mode:               direct | server          (default: direct)
    memalpha_repo_path: Mem-alpha                (path relative to project root)
    agent_config_path:  Mem-alpha/config/memalpha-qwen3-4b_agent_0.05-0.1.yaml
    chunk_size:         2000                     (chars per context chunk)
    max_new_tokens:     2048
    # server mode only:
    memory_server_url:  http://localhost:5005     (base URL of memory_server.py)
"""

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml

log = logging.getLogger(__name__)

# Chunk prompt matching Mem-alpha's unified_prompt (prompts_wrt_datasource.yaml)
_CHUNK_PROMPT = (
    "Remember the following content chunk by completing these steps:\n\n"
    "1. **Core Memory Update**: Maintain an understanding of the user, or a summary "
    "of what the user is reading, or a set of classification rules summarized from "
    "the classification examples. Keep updates brief (a few sentences maximum).\n\n"
    "2. **Memory Storage**:\n"
    "   - **Episodic Memory**: Record user actions and key events with timestamps "
    "(format: \"At timestamp t, user did X\")\n"
    "   - **Semantic Memory**: Record key facts and information "
    "(format: \"John is User's 18-year-old friend\", \"Harry Potter author: J.K. Rowling\")\n\n"
    "<new_chunk>\n{context}\n</new_chunk>\n\n"
    "**Important**: Response limit is {max_new_tokens} tokens. "
    "Be concise and brief in all memory updates."
)


def _clean_answer(text: str) -> str:
    """Strip <think>...</think> reasoning and return only the final answer."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


class MemAlphaAdapter:
    """ModelAdapter for YuWangX/Memalpha-4B using Mem-alpha's agent + memory pipeline."""

    def __init__(self):
        self._agent = None
        self._Memory = None          # Memory class, imported from Mem-alpha
        self._agent_config: Dict[str, Any] = {}
        self._cfg: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # ModelAdapter interface
    # ------------------------------------------------------------------

    def load(self, config: Dict[str, Any]) -> None:
        self._cfg = config

        repo_path = Path(config.get("memalpha_repo_path", "Mem-alpha"))
        if not repo_path.is_absolute():
            repo_path = Path(__file__).parent.parent / repo_path
        repo_path = repo_path.resolve()

        if not repo_path.exists():
            raise FileNotFoundError(
                f"Mem-alpha repo not found at {repo_path}. "
                "Clone it with: git clone https://github.com/wangyu-ustc/Mem-alpha Mem-alpha"
            )

        # Add Mem-alpha to sys.path so its internal imports resolve
        repo_str = str(repo_path)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        # Load agent YAML config
        agent_cfg_path = config.get("agent_config_path")
        if agent_cfg_path:
            p = Path(agent_cfg_path)
            if not p.is_absolute():
                p = Path(__file__).parent.parent / p
            with open(p) as f:
                self._agent_config = yaml.safe_load(f)
        else:
            # Minimal defaults matching the 4B config
            self._agent_config = {
                "agent_name": "memalpha_bench",
                "model_name": config.get("model_name", "YuWangX/Memalpha-4B"),
                "enable_thinking": config.get("enable_thinking", False),
                "vllm": config.get("vllm", True),
                "max_new_tokens": config.get("max_new_tokens", 2048),
                "infer_with_full_memory": False,  # direct mode default
            }

        # In direct mode, force off external-model routing
        if config.get("mode", "direct") == "direct":
            self._agent_config["infer_with_full_memory"] = False
            self._agent_config.pop("external_model_url", None)

        log.info("Loading Mem-alpha MemoryAgent (model=%s)", self._agent_config.get("model_name"))
        from agent import MemoryAgent  # noqa: E402  (from Mem-alpha repo)
        from memory import Memory      # noqa: E402

        self._Memory = Memory
        self._agent = MemoryAgent(agent_config=self._agent_config)
        log.info("MemAlpha adapter ready (mode=%s)", config.get("mode", "direct"))

    def predict(self, context: str, question: str) -> str:
        """Build memories from context, then answer the question."""
        self._reset_memory()

        chunks = self._split_context(context)
        max_new = self._agent_config.get("max_new_tokens", 2048)
        budget = max(int(max_new * 0.8), 256)

        log.debug("Memory building: %d chunks", len(chunks))
        for chunk in chunks:
            prompt = _CHUNK_PROMPT.format(context=chunk, max_new_tokens=budget)
            self._agent.chat(prompt, status="memorie")

        mode = self._cfg.get("mode", "direct")
        if mode == "server":
            return self._answer_via_server(question)
        else:
            raw = self._agent.chat(question, status="chat")
            return _clean_answer(raw)

    def teardown(self) -> None:
        # vLLM holds GPU memory; nothing to explicitly release via the public API
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_memory(self) -> None:
        """Fresh memory state for each new example."""
        including_core = self._agent_config.get("including_core", False)
        self._agent.memory = self._Memory(including_core=including_core)
        self._agent.conversation_history = []
        self._agent.step = 0

    def _split_context(self, context: str) -> List[str]:
        """
        Split context into chunks for sequential memory building.
        Prefers document boundaries ("Document N:") when present;
        falls back to character-based splitting.
        """
        chunk_size: int = self._cfg.get("chunk_size", 2000)

        # Split on document markers (produced by our dataset adapters)
        parts = re.split(r"(?=Document \d+:)", context)
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) <= 1:
            # No document markers — split by characters at paragraph boundaries
            return self._char_split(context, chunk_size)

        # Merge small adjacent docs into chunk_size buckets
        chunks, current, current_len = [], [], 0
        for part in parts:
            if current and current_len + len(part) > chunk_size:
                chunks.append("\n\n".join(current))
                current, current_len = [], 0
            current.append(part)
            current_len += len(part)
        if current:
            chunks.append("\n\n".join(current))
        return chunks

    @staticmethod
    def _char_split(text: str, chunk_size: int) -> List[str]:
        """Split text at paragraph boundaries into ~chunk_size character chunks."""
        paragraphs = text.split("\n\n")
        chunks, current, current_len = [], [], 0
        for para in paragraphs:
            if current and current_len + len(para) > chunk_size:
                chunks.append("\n\n".join(current))
                current, current_len = [], 0
            current.append(para)
            current_len += len(para)
        if current:
            chunks.append("\n\n".join(current))
        return chunks or [text]

    def _answer_via_server(self, question: str) -> str:
        """POST memories + question to memory_server.py /batch_process."""
        base_url = self._cfg.get("memory_server_url", "http://localhost:5005")
        endpoint = base_url.rstrip("/") + "/batch_process"

        memory = self._agent.memory
        memory_dict: Dict[str, Any] = {
            "episodic": memory.episodic,
            "semantic": memory.semantic,
        }
        if memory.including_core and memory.core is not None:
            memory_dict["core"] = memory.core

        payload = {
            "memories": [memory_dict],
            "questions": [[question]],
        }

        try:
            resp = requests.post(endpoint, json=payload, timeout=120)
            resp.raise_for_status()
            result = resp.json().get("result", [[]])[0]
            raw = result[0] if result else ""
        except requests.RequestException as e:
            raise RuntimeError(
                f"Memory server at {endpoint} failed: {e}. "
                "Start it with: cd Mem-alpha && python memory_server.py "
                "--server_url http://localhost:8001/v1 --port 5005"
            ) from e

        return _clean_answer(raw)
