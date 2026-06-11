"""
models/mem1.py — Adapter for MIT-MI/MEM1 (Mem-Lab/Qwen2.5-7B-RL-RAG-Q2-EM-Release)

MEM1 is an RL-trained Qwen2.5-7B agent that operates with CONSTANT MEMORY across
a multi-turn <think>/<search>/<answer> loop (up to MAX_ITERATION=6 turns). At
each turn it produces a <think>...</think> summary, then either issues
<search>query</search> or a final <answer>...</answer>; the entire prior
observation is then DISCARDED ("constant memory") and replaced with just the
current turn's response + the new search results.

IMPORTANT — this is an adaptation, not a faithful reproduction
-----------------------------------------------------------------
MEM1's own evaluation has NO `context` field at all: <search> queries hit a
live HTTP retrieval server (SEARCH_URL) backed by the full Wikipedia-18 corpus
via FAISS/pyserini. That is fundamentally incompatible with this benchmark's
"(context, question) -> answer" interface (same issue as Agent-Lightning's RAG
example and MemSearcher).

To make MEM1 runnable here, `<search>` is redirected to a PER-EXAMPLE BM25
index built on the fly over the example's `context` field (split into passages
the same way models/memalpha.py splits multi-document contexts), instead of a
Wikipedia-scale corpus. Everything else (prompt template, constant-memory loop,
stop-sequence-driven multi-turn completion, hint text, answer extraction) is
copied verbatim from Mem1/inference/{data_pipelines.py,models.py} and
Mem1/gen_data/data_process/nq_search.py.

Practical consequence: with only a handful of passages per example, BM25
top-k often retrieves most of the relevant context in 1-2 turns, so the
6-turn loop may rarely be fully exercised on short-context datasets
(HotpotQA/MuSiQue/2WikiMultihopQA). It will be exercised more on datasets with
many passages (QASPER, NarrativeQA full_text, RULER).

Repo:    https://github.com/MIT-MI/MEM1
Paper:   https://arxiv.org/abs/2506.15841
Weights: Mem-Lab/Qwen2.5-7B-RL-RAG-Q2-EM-Release  (Qwen2.5-7B based, ~15GB bf16)

Requires an OpenAI-compatible server (vLLM) exposing the raw /v1/completions
endpoint with vLLM's `stop_reason` extension:
    vllm serve Mem-Lab/Qwen2.5-7B-RL-RAG-Q2-EM-Release \\
        --max-model-len 8192 --gpu-memory-utilization 0.6 --port 8000

Config keys (all under cfg["model"]):
    adapter:              mem1
    mode:                 local | remote        (default: remote)
    model_name_or_path:   Mem-Lab/Qwen2.5-7B-RL-RAG-Q2-EM-Release
    tokenizer_path:       Qwen/Qwen2.5-7B        (base tokenizer, for apply_chat_template)
    base_url:             http://localhost:8000/v1
    # ---- local mode only (ignored when mode: remote) ----
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.6
    max_model_len:        8192
    # ---- MEM1 agent-loop parameters (match the official inference code) ----
    max_iterations:       6        # MAX_ITERATION
    top_k:                3        # TOP_K passages returned per <search>
    max_new_tokens:       1024     # per-turn generation budget
    temperature:          0.01
    top_p:                0.95
    request_timeout:      300
"""

import logging
import os
import re
import subprocess
import time
from typing import Any, Dict, List, Optional

import requests

log = logging.getLogger(__name__)

MAX_ITERATION = 6
TOP_K = 3

# ---------------------------------------------------------------------------
# Prompt template — verbatim from Mem1/gen_data/data_process/nq_search.py
# (make_prefix, template_type='base')
# ---------------------------------------------------------------------------

_PREFIX_TEMPLATE = """You will answer complex questions through iterative summary and web search.

Your response must include:

<think>
- Keep information from the current information that is potentially relevant and useful for answering the question.
- The current information will be discarded in the next step and the think part will be the only information you have to complete the task.
- You should also summarize previous searches you have made to avoid repetitive searches.
- You will be told how many turns you have left inside the information given to you after you have made a search. You should keep track of the number of turns you have left.
</think>

Then either:
<search>
QUERY (only if you have turns left)
</search>

Or:
<answer>
FINAL ANSWER ONLY (no explanation)
</answer>

Follow this format strictly for your response so that it's either <think>...</think><search>...</search> or <think>...</think><answer>...</answer>.

Question: {question}
"""

_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _parse_action(response: str) -> Optional[Dict[str, str]]:
    """Mirrors data_pipelines.py::act() — returns the search query or final answer."""
    if "<search>" in response and "</search>" in response:
        m = _SEARCH_RE.search(response)
        return {"type": "search", "query": (m.group(1).strip() if m else "")}
    if "<answer>" in response and "</answer>" in response:
        m = _ANSWER_RE.search(response)
        return {"type": "answer", "content": (m.group(1).strip() if m else "")}
    return None


# ---------------------------------------------------------------------------
# BM25-over-context retrieval — replaces MEM1's Wikipedia/FlashRAG SEARCH_URL
# ---------------------------------------------------------------------------

_DOC_SPLIT_RE = re.compile(r"(?=Document \d+:)")
_DOC_PREFIX_RE = re.compile(r"Document\s+\d+:\s*(.*)", re.DOTALL)


def _split_context_to_passages(context: str) -> List[Dict[str, str]]:
    """Split `context` into {title, text} passages for per-example BM25 search.

    Mirrors the "Title\\nText" structure of FlashRAG corpus entries that
    batch_search()/passages2string() in the original MEM1 code expect.
    Prefers "Document N: <Title>\\n<text>" markers (as produced by this
    project's multi-document dataset adapters, e.g. memalpha._split_context);
    falls back to paragraph-based chunks with generic titles.
    """
    context = (context or "").strip()
    if not context:
        return []

    parts = [p.strip() for p in _DOC_SPLIT_RE.split(context) if p.strip()]

    if parts and parts[0].startswith("Document "):
        passages = []
        for part in parts:
            m = _DOC_PREFIX_RE.match(part)
            body = m.group(1).strip() if m else part
            head, _, rest = body.partition("\n")
            title = head.strip() or "Untitled"
            text = rest.strip() or title
            passages.append({"title": title, "text": text})
        return passages

    # Fallback: paragraph-based chunks with generic titles
    paragraphs = [p.strip() for p in context.split("\n\n") if p.strip()] or [context]
    return [{"title": f"Passage {i + 1}", "text": p} for i, p in enumerate(paragraphs)]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


class _ContextRetriever:
    """Per-example BM25 index over `context`'s passages (TOP_K per query)."""

    def __init__(self, context: str, top_k: int = TOP_K):
        self.passages = _split_context_to_passages(context)
        self.top_k = top_k
        self._bm25 = None
        if self.passages:
            try:
                from rank_bm25 import BM25Plus
            except ImportError as e:
                raise ImportError(
                    "models/mem1.py requires rank_bm25 for its per-example "
                    "search index. Install with: pip install rank_bm25"
                ) from e
            # BM25Plus (not BM25Okapi): with the small per-example corpora typical
            # here (e.g. 2-10 documents for HotpotQA/MuSiQue), BM25Okapi's IDF term
            # is ~0 or negative for most words, making most documents score
            # identically. BM25Plus's IDF stays positive regardless of corpus size.
            corpus = [_tokenize(f"{p['title']} {p['text']}") for p in self.passages]
            self._bm25 = BM25Plus(corpus)

    def search(self, query: str) -> str:
        """Return a 'Doc N(Title: ...) text' block, mirroring passages2string()."""
        if not self._bm25 or not self.passages:
            return "No documents found.\n"
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(self.passages)), key=lambda i: scores[i], reverse=True)
        out = ""
        for rank, idx in enumerate(ranked[: self.top_k]):
            p = self.passages[idx]
            out += f"Doc {rank + 1}(Title: {p['title']}) {p['text']}\n"
        return out


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class Mem1Adapter:
    """ModelAdapter for MIT-MI/MEM1 with BM25-over-context retrieval."""

    def __init__(self):
        self._tokenizer = None
        self._server_proc: Optional[subprocess.Popen] = None
        self._cfg: Dict[str, Any] = {}
        self._base_url = ""

    # ------------------------------------------------------------------
    # ModelAdapter interface
    # ------------------------------------------------------------------

    def load(self, config: Dict[str, Any]) -> None:
        self._cfg = config
        mode = config.get("mode", "remote")

        tokenizer_path = config.get("tokenizer_path", "Qwen/Qwen2.5-7B")
        log.info("Loading tokenizer from %s", tokenizer_path)
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        if mode == "local":
            self._start_local_server(config)
        else:
            log.info("Connecting to remote endpoint at %s", config.get("base_url"))

        self._base_url = config.get("base_url", "http://localhost:8000/v1").rstrip("/")
        log.info("MEM1 adapter ready (mode=%s)", mode)

    def predict(self, context: str, question: str) -> str:
        """MEM1's constant-memory <think>/<search>/<answer> loop, with <search>
        served by a per-example BM25 index over `context` instead of SEARCH_URL."""
        max_iterations = self._cfg.get("max_iterations", MAX_ITERATION)
        top_k = self._cfg.get("top_k", TOP_K)
        retriever = _ContextRetriever(context, top_k=top_k)

        prompt = _PREFIX_TEMPLATE.format(question=question.strip())
        cur_obs = ""

        for iteration in range(max_iterations):
            is_last_turn = iteration == max_iterations - 1
            cur_response = self._make_completion(prompt, cur_obs, is_last_turn)
            if cur_response is None:
                return ""

            # Constant memory: discard everything accumulated so far.
            cur_obs = ""

            action = _parse_action(cur_response)
            if action is None:
                return ""

            if action["type"] == "answer":
                return action["content"]

            # action["type"] == "search"
            num_turns_left = max_iterations - iteration - 1
            if num_turns_left > 1:
                hint = f"[HINT]You have {num_turns_left} turns left.[/HINT]"
            else:
                hint = f"[HINT]You have {num_turns_left} turn left. You must answer the question now.[/HINT]"

            search_results = retriever.search(action["query"])
            info_block = f"<information>\n{hint}\n{search_results}\n</information>"
            cur_obs = cur_response + info_block

        return ""  # exhausted max_iterations without an <answer>

    def teardown(self) -> None:
        if self._server_proc is not None:
            log.info("Stopping local vLLM server (pid=%d)…", self._server_proc.pid)
            self._server_proc.terminate()
            self._server_proc = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_completion(self, initial_prompt: str, content: str, is_last_turn: bool) -> Optional[str]:
        """Mirrors models.py::VLLMOpenAIClient.make_completion()."""
        messages = [
            {"role": "user", "content": initial_prompt},
            {"role": "assistant", "content": content},
        ]
        prompt_text = self._tokenizer.apply_chat_template(messages, tokenize=False)
        # Strip the trailing "<|im_end|>\n" so generation continues this assistant turn.
        suffix = "<|im_end|>\n"
        if prompt_text.endswith(suffix):
            prompt_text = prompt_text[: -len(suffix)]

        stop = ["</answer>"] if is_last_turn else ["</search>", "</answer>"]
        model = self._cfg.get("model_name_or_path", "Mem-Lab/Qwen2.5-7B-RL-RAG-Q2-EM-Release")

        try:
            resp = requests.post(
                f"{self._base_url}/completions",
                json={
                    "model": model,
                    "prompt": prompt_text,
                    "temperature": self._cfg.get("temperature", 0.01),
                    "top_p": self._cfg.get("top_p", 0.95),
                    "top_k": -1,
                    "max_tokens": self._cfg.get("max_new_tokens", 1024),
                    "stop": stop,
                },
                timeout=self._cfg.get("request_timeout", 300),
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            log.warning("MEM1 completion request failed: %s", e)
            return None

        choice = resp.json()["choices"][0]
        text = choice["text"].strip()
        stop_reason = choice.get("stop_reason")
        if stop_reason == "</search>":
            text += "</search>"
        elif stop_reason == "</answer>":
            text += "</answer>"
        return text

    def _start_local_server(self, config: Dict[str, Any]) -> None:
        model = config.get("model_name_or_path", "Mem-Lab/Qwen2.5-7B-RL-RAG-Q2-EM-Release")
        tp = config.get("tensor_parallel_size", 1)
        gpu_mem = config.get("gpu_memory_utilization", 0.6)
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
