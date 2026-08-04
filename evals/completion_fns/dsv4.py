"""A CompletionFn for the 8-rank DeepSeek-V4-Flash-0731 ONNX export.

The model is tensor/expert parallel across every GPU on the node, so unlike
`ort_genai.py` it cannot be sharded one worker per GPU: a single `DSV4Engine`
owns all eight ranks and answers requests serially.  The engine is therefore a
process-wide singleton, created on first use and torn down at exit.

Prompt encoding comes from the checkpoint's own `encoding/encoding_dsv4.py`
(this release ships no Jinja chat template), via the `dsv4_prompt` helper in the
onnxruntime-genai example directory.

Usage::

    OPENAI_API_KEY=dummy EVALS_THREADS=1 \
      oaieval dsv4/flash-0731-mcq10 match_mmlu_pro_800
"""

import atexit
import logging
import os
import queue
import re
import sys
import threading
import time
from typing import Any, Optional, Union

from evals.api import CompletionFn, CompletionResult
from evals.prompt.base import OpenAICreateChatPrompt, Prompt, is_chat_prompt
from evals.record import record_sampling

logger = logging.getLogger(__name__)

DEFAULT_EXAMPLE_DIR = os.path.expanduser(
    "~/git/onnxruntime-genai/examples/python/deepseek_v4_flash_0731")

_engine = None
_engine_key = None
_engine_lock = threading.Lock()
_batcher = None


def _import_helpers(example_dir: str):
    example_dir = os.path.abspath(os.path.expanduser(example_dir))
    if example_dir not in sys.path:
        sys.path.insert(0, example_dir)
    import dsv4_engine
    import dsv4_prompt

    return dsv4_engine, dsv4_prompt


def _get_engine(dsv4_engine, model_path: str, world: int, port: int, log_dir: str,
                batch: int):
    """One engine per process; it holds every GPU it was given."""
    global _engine, _engine_key
    key = (model_path, world, port, batch)
    with _engine_lock:
        if _engine is not None and _engine_key != key:
            raise RuntimeError(
                f"a DSV4 engine for {_engine_key} is already running in this process; "
                f"only one multi-GPU model can be resident at a time")
        if _engine is None:
            _engine = dsv4_engine.DSV4Engine(model_path, world=world, port=port,
                                             log_dir=log_dir, batch=batch,
                                             cuda_graph=bool(os.environ.get("DSV4_CUDA_GRAPH")))
            _engine_key = key
            atexit.register(_shutdown)
    return _engine


def _shutdown():
    global _engine
    if _engine is not None:
        _engine.close()
        _engine = None


class _Batcher:
    """Collects concurrent single-prompt calls into one engine batch.

    oaieval hands out one sample per worker thread, but a decode step costs the
    same whatever the rows contain -- it is ~18k kernel launches either way -- so
    waiting a moment for the sibling threads to arrive is nearly free and divides
    the wall time by the batch size.  Requires `EVALS_THREADS` >= `batch`.
    """

    def __init__(self, engine, batch: int, wait_s: float = 1.0):
        self.engine = engine
        self.batch = batch
        self.wait_s = wait_s
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, tokens, max_new_tokens, eos_token_ids):
        item = {"tokens": tokens, "max": max_new_tokens, "eos": eos_token_ids,
                "done": threading.Event(), "out": None, "err": None}
        self.q.put(item)
        item["done"].wait()
        if item["err"] is not None:
            raise item["err"]
        return item["out"]

    def _drain(self):
        items = [self.q.get()]
        deadline = time.time() + self.wait_s
        while len(items) < self.batch:
            left = deadline - time.time()
            if left <= 0:
                break
            try:
                items.append(self.q.get(timeout=left))
            except queue.Empty:
                break
        return items

    def _run(self):
        while True:
            items = self._drain()
            try:
                out = self.engine.generate_batch(
                    [i["tokens"] for i in items],
                    max_new_tokens=max(i["max"] for i in items),
                    eos_token_ids=items[0]["eos"])
                for b, item in enumerate(items):
                    item["out"] = {"tokens": out["tokens"][b],
                                   "stop_reason": out["stop_reason"][b]}
            except Exception as exc:  # hand the failure back to every waiter
                for item in items:
                    item["err"] = exc
            finally:
                for item in items:
                    item["done"].set()


def extract_choice_letter(text: str, choice_letters: str = "ABCD") -> str:
    """Pull a multiple-choice letter out of the model's answer.

    Preference order matches `ort_genai.extract_choice_letter`: an explicit
    answer marker, then a bracketed option, then the last standalone letter.
    """
    if not text:
        return ""
    cls = "".join(dict.fromkeys(choice_letters.upper()))
    marker = re.findall(
        rf"(?:^|\n|\r)\s*(?:final\s+answer|answer|correct\s+answer)\s*[:\-]?\s*([{cls}])\b",
        text, re.IGNORECASE)
    if marker:
        return marker[-1].upper()
    bracket = re.findall(rf"\[\s*([{cls}])\s*\]", text, re.IGNORECASE)
    if bracket:
        return bracket[-1].upper()
    tokens = re.findall(rf"\b([{cls}])\b", text, re.IGNORECASE)
    return tokens[-1].upper() if tokens else ""


class DSV4CompletionResult(CompletionResult):
    def __init__(self, completion: str, raw: str, prompt: Any):
        self.completion = completion
        self.raw = raw
        self.prompt = prompt

    def get_completions(self) -> list[str]:
        return [self.completion.strip()]


class DSV4CompletionFn(CompletionFn):
    def __init__(
        self,
        model_path: str,
        checkpoint: str,
        world: int = 8,
        port: int = 19555,
        log_dir: str = "/tmp",
        example_dir: Optional[str] = None,
        max_new_tokens: int = 64,
        thinking_mode: str = "chat",
        reasoning_effort: str = "low",
        extract_letter_choice: bool = False,
        choice_letters: str = "ABCD",
        system_prompt: Optional[str] = None,
        model_name: str = "deepseek-v4-flash-0731",
        batch: int = 1,
        **_,
    ):
        example_dir = example_dir or os.environ.get("DSV4_EXAMPLE_DIR", DEFAULT_EXAMPLE_DIR)
        dsv4_engine, dsv4_prompt = _import_helpers(example_dir)

        batch = int(os.environ.get("DSV4_EVAL_BATCH", batch))
        self.max_new_tokens = max_new_tokens
        self.extract_letter_choice = extract_letter_choice
        self.choice_letters = choice_letters
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.thinking_mode = thinking_mode

        self.prompt = dsv4_prompt.DSV4Prompt(checkpoint, thinking_mode, reasoning_effort)
        self.engine = _get_engine(dsv4_engine, model_path, world, port, log_dir, batch)
        global _batcher
        with _engine_lock:
            if _batcher is None:
                _batcher = _Batcher(self.engine, batch)
        self.batcher = _batcher

    def _build_messages(self, prompt: Union[str, OpenAICreateChatPrompt]) -> list[dict]:
        if is_chat_prompt(prompt):
            messages = [dict(msg) for msg in prompt]
        else:
            messages = [{"role": "user", "content": str(prompt)}]
        if self.system_prompt and not any(m.get("role") == "system" for m in messages):
            messages = [{"role": "system", "content": self.system_prompt}] + messages
        return messages

    def __call__(self, prompt: Union[str, OpenAICreateChatPrompt], **kwargs) -> DSV4CompletionResult:
        if isinstance(prompt, Prompt):
            prompt = prompt.to_formatted_prompt()

        messages = self._build_messages(prompt)
        full_prompt = self.prompt.render(messages)
        input_tokens = self.prompt.encode(full_prompt)

        out = self.batcher.submit(input_tokens, self.max_new_tokens,
                                  self.prompt.eos_token_ids)

        raw = self.prompt.decode(out["tokens"])
        answer = self.prompt.parse(raw)["content"] or raw
        if self.extract_letter_choice:
            extracted = extract_choice_letter(answer, self.choice_letters)
            if extracted:
                answer = extracted

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("RAW completion (%s): %r", out["stop_reason"], raw)
            logger.debug("FINAL answer: %r", answer)

        record_sampling(prompt=full_prompt, sampled=[answer], model=self.model_name)
        return DSV4CompletionResult(completion=answer, raw=raw, prompt=full_prompt)
