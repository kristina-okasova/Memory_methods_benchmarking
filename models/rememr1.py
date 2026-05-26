"""
models/rememr1.py — Adapter for yrshi/ReMemR1-7B

Replicates the evaluation inference loop from taskutils/memory_eval/utils/rememr1.py:
  - Token-based chunking (chunk_size = 5000 tokens)
  - Center-crop truncation at max_context_len tokens
  - Per-chunk memory update with <update> / <recall> parsing
  - TF-IDF retrieval of previously written memories on recall
  - Final answer in \\boxed{} format

Requires an OpenAI-compatible server (SGLang recommended):
  python -m sglang.launch_server \\
      --model-path yrshi/ReMemR1-7B \\
      --served-model-name yrshi/ReMemR1-7B \\
      --port 8000

Config keys (all under cfg["model"]):
    adapter:            rememr1
    model_name_or_path: yrshi/ReMemR1-7B     # also used as the API model name
    base_url:           http://localhost:8000/v1
    api_key:            123-abc
    chunk_size:         5000
    max_context_len:    120000
    max_new_tokens:     1024
    temperature:        0.7    # matches training default
    top_p:              0.95
"""

import logging
import os
import re
from typing import Any, Dict, Optional

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Templates — verbatim from ReMemR1/taskutils/memory_eval/utils/rememr1.py
# ---------------------------------------------------------------------------

_TEMPLATE = (
    "You are presented with a problem, a section of an article that may contain "
    "the answer to the problem, and a previous memory. You should generate response "
    "in the following format:\n"
    "- Output your thinking process in <thinking>your_thinking_process</thinking>.\n"
    "- Read the provided section carefully and update the memory with the new "
    "information that helps to answer the problem in only one "
    "<update>the_updated_memory</update> action. Be sure to retain all relevant "
    "details from the previous memory while adding any new, useful information.\n"
    "- If you notice partial key evidence that is not enough to answer the problem, "
    "also output only one `<recall>query</recall>` "
    "(e.g. `<recall>who's the president of the United States?</recall>`) to retrieve "
    "information in previous memories.\n\n"
    "<problem> \n{prompt}\n</problem>\n\n"
    "<recalled_memory>\n{recalled_memory}\n</recalled_memory>\n\n"
    "<memory>\n{memory}\n</memory>\n\n"
    "<section>\n{chunk}\n</section>\n\n"
    "Updated memory:\n"
)

_TEMPLATE_FINAL = (
    "You are presented with a problem and a previous memory. Please answer the "
    "problem based on the previous memory and put the answer in \\boxed{{}}.\n\n"
    "<problem> \n{prompt}\n</problem>\n\n"
    "<recalled_memory>\n{recalled_memory}\n</recalled_memory>\n\n"
    "<memory>\n{memory}\n</memory>\n\n"
    "Your answer:\n"
)

_NO_MEMORY = "No previous memory"
_NO_RECALLED_MEMORY = "No memory was recalled."


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_recall_query(text: str) -> Optional[str]:
    m = re.search(r"<recall>(.+)</recall>", text)
    return m.group(1) if m else None


def _parse_update_memory(text: str) -> str:
    """Strip <recall> tags; keep the rest as the updated memory string."""
    cleaned = re.sub(r"<recall>.*?</recall>", "", text, flags=re.DOTALL)
    return cleaned.strip() or _NO_MEMORY


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


# ---------------------------------------------------------------------------
# TF-IDF retriever (mirrors ReMemR1/taskutils/memory_eval/utils/tf_idf_retriever.py)
# ---------------------------------------------------------------------------

class _TfidfRetriever:
    def __init__(self, tokenizer):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._tokenizer = tokenizer
        self._vectorizer = TfidfVectorizer(tokenizer=self._tok)

    def _tok(self, text: str):
        tokens = self._tokenizer.tokenize(text.lower())
        return [t.replace("Ġ", "") for t in tokens]

    def top1(self, query: str, corpus: set) -> Optional[str]:
        from sklearn.metrics.pairwise import cosine_similarity
        if not query or not corpus:
            return None
        docs = list(corpus)
        try:
            mat = self._vectorizer.fit_transform(docs)
            qvec = self._vectorizer.transform([query])
            sims = cosine_similarity(qvec, mat).flatten()
            return docs[int(np.argmax(sims))]
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class ReMemR1Adapter:
    """ModelAdapter wrapping ReMemR1 served via OpenAI-compatible API (SGLang/vLLM)."""

    def __init__(self):
        self._client = None
        self._tokenizer = None
        self._retriever = None
        self._cfg: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # ModelAdapter interface
    # ------------------------------------------------------------------

    def load(self, config: Dict[str, Any]) -> None:
        self._cfg = config
        model_path = config["model_name_or_path"]

        log.info("Loading tokenizer from %s", model_path)
        self._tokenizer = self._load_tokenizer(model_path)
        self._retriever = _TfidfRetriever(self._tokenizer)

        base_url = config.get("base_url", "http://localhost:8000/v1")
        api_key = config.get("api_key") or os.environ.get("REMEMR1_API_KEY", "123-abc")
        self._client = self._build_client(base_url, api_key)
        log.info("ReMemR1 adapter ready — endpoint: %s", base_url)

    def predict(self, context: str, question: str) -> str:
        """Recurrent memory-update loop with TF-IDF recall, then boxed answer."""
        max_ctx = self._cfg.get("max_context_len", 120000)
        chunk_size = self._cfg.get("chunk_size", 5000)

        input_ids = self._tokenizer.encode(context)
        if len(input_ids) > max_ctx:
            half = max_ctx // 2
            input_ids = input_ids[:half] + input_ids[-half:]

        memory = _NO_MEMORY
        recalled_memory = _NO_RECALLED_MEMORY
        history: set = set()

        for i in range(0, max(1, len(input_ids)), chunk_size):
            chunk_text = self._tokenizer.decode(
                input_ids[i : i + chunk_size], skip_special_tokens=True
            )
            msg = _TEMPLATE.format(
                prompt=question,
                chunk=chunk_text,
                memory=memory,
                recalled_memory=recalled_memory,
            )
            raw = self._call_llm(msg)
            if raw is None:
                return ""

            memory = _parse_update_memory(raw)
            history.add(memory)

            query = _parse_recall_query(raw)
            if query:
                hit = self._retriever.top1(query, history)
                recalled_memory = hit if hit else _NO_RECALLED_MEMORY
            else:
                recalled_memory = _NO_RECALLED_MEMORY

        final_msg = _TEMPLATE_FINAL.format(
            prompt=question,
            memory=memory,
            recalled_memory=recalled_memory,
        )
        raw = self._call_llm(final_msg)
        if not raw:
            return ""
        return (_extract_boxed(raw) or raw).strip()

    def teardown(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_llm(self, user_content: str) -> Optional[str]:
        max_tokens = self._cfg.get("max_new_tokens", 1024)
        temperature = self._cfg.get("temperature", 0.7)
        top_p = self._cfg.get("top_p", 0.95)
        model = self._cfg.get("model_name_or_path", "yrshi/ReMemR1-7B")
        try:
            resp = self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_content}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            log.error("ReMemR1 LLM call failed: %s", e)
            return None

    @staticmethod
    def _load_tokenizer(model_name_or_path: str):
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError("pip install transformers") from e
        return AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    @staticmethod
    def _build_client(base_url: str, api_key: str):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("pip install openai>=1.0") from e
        return OpenAI(base_url=base_url, api_key=api_key)
