"""
A CompletionFn that runs a local ONNX model in-process with onnxruntime-genai.

This mirrors the in-process generation approach used by the onnxruntime-genai
benchmark scripts (e.g. benchmark_e2e.py / model-chat.py): it loads an ONNX
model folder once, applies the model's chat template, and greedily (by default)
decodes a response.

It was written to evaluate gpt-oss-20b, whose output uses the "harmony" format
with multiple channels (analysis / commentary / final). Only the contents of the
`final` channel are returned as the answer; see `extract_final_channel`.
"""
import json
import logging
import os
import re
import threading
from typing import Any, Optional, Union

import onnxruntime_genai as og

from evals.api import CompletionFn, CompletionResult
from evals.prompt.base import (
    CompletionPrompt,
    OpenAICreateChatPrompt,
    Prompt,
    is_chat_prompt,
)
from evals.record import record_sampling

logger = logging.getLogger(__name__)


# Harmony control markers emitted by gpt-oss style models.
_FINAL_MARKER = "<|channel|>final<|message|>"
_MESSAGE_MARKER = "<|message|>"
# Tokens that terminate a message / that should never appear in a clean answer.
_TERMINATORS = ("<|return|>", "<|end|>", "<|call|>")
_CONTROL_TOKENS = (
    "<|start|>",
    "<|end|>",
    "<|return|>",
    "<|call|>",
    "<|channel|>",
    "<|message|>",
    "<|constrain|>",
)

# Some models emit non-breaking / narrow / zero-width spaces (e.g. U+202F, U+00A0)
# which break naive substring matching in string-based evals even when the answer
# is otherwise correct. Normalize them to plain spaces (or drop zero-width ones).
_WHITESPACE_TRANSLATION = {
    0x00A0: " ",  # no-break space
    0x202F: " ",  # narrow no-break space
    0x2007: " ",  # figure space
    0x2009: " ",  # thin space
    0x200A: " ",  # hair space
    0x200B: "",  # zero-width space
    0xFEFF: "",  # zero-width no-break space / BOM
}

_LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
_BRACKET_LETTER_RE = re.compile(r"\[\s*([ABCD])\s*\]", re.IGNORECASE)
_ANSWER_MARKER_RE = re.compile(
    r"(?:^|\n|\r)\s*(?:final\s+answer|answer|correct\s+answer)\s*[:\-]?\s*([ABCD])\b",
    re.IGNORECASE,
)


def extract_final_channel(text: str) -> str:
    """Extract the user-facing answer from a harmony-formatted generation.

    gpt-oss emits something like::

        <|channel|>analysis<|message|>...reasoning...<|end|>
        <|start|>assistant<|channel|>final<|message|>...answer...<|return|>

    We want only the contents of the last ``final`` channel. If the model did not
    emit a final channel (e.g. a truncated or looping generation), we fall back to
    the contents of the last channel block, with stray control tokens stripped.
    """
    answer = text

    if _FINAL_MARKER in answer:
        # Take everything after the last final-channel marker.
        answer = answer.rsplit(_FINAL_MARKER, 1)[-1]
    elif "final" + _MESSAGE_MARKER in answer:
        # Markers partially stripped, but "final<|message|>" survived.
        answer = answer.rsplit("final" + _MESSAGE_MARKER, 1)[-1]
    elif _MESSAGE_MARKER in answer:
        # No final channel: fall back to the content of the last channel block
        # rather than gluing the channel name onto the text.
        answer = answer.rsplit(_MESSAGE_MARKER, 1)[-1]

    # Cut off anything after the message terminator.
    for term in _TERMINATORS:
        idx = answer.find(term)
        if idx != -1:
            answer = answer[:idx]

    # Remove any residual control tokens.
    for tok in _CONTROL_TOKENS:
        answer = answer.replace(tok, "")

    # Normalize unicode whitespace artifacts to plain ASCII spaces.
    answer = answer.translate(_WHITESPACE_TRANSLATION)

    return answer.strip()


def extract_choice_letter(text: str) -> str:
    """Extract a final A/B/C/D choice from model text.

    Order of preference mirrors OpenAIChatToCompletionLetterFn:
    explicit answer markers, bracketed option form, then last standalone token.
    """
    if not text:
        return ""

    marker = _ANSWER_MARKER_RE.findall(text)
    if marker:
        return marker[-1].upper()

    bracket = _BRACKET_LETTER_RE.findall(text)
    if bracket:
        return bracket[-1].upper()

    tokens = _LETTER_RE.findall(text)
    return tokens[-1].upper() if tokens else ""


class ORTGenAICompletionResult(CompletionResult):
    def __init__(self, completion: str, raw: str, prompt: Any):
        self.completion = completion
        self.raw = raw
        self.prompt = prompt

    def get_completions(self) -> list[str]:
        return [self.completion]


class ORTGenAICompletionFn(CompletionFn):
    """Generate completions from a local ONNX model via onnxruntime-genai.

    Args:
        model_path: Path to the ONNX model folder (contains genai_config.json).
        execution_provider: "cuda" (default), "cpu", or "follow_config" to use
            whatever provider is stored in genai_config.json.
        max_new_tokens: Maximum number of tokens to generate per sample.
        do_sample: If False (default), decode greedily (deterministic).
        temperature/top_p/top_k: Sampling options (only used when do_sample).
        system_prompt: Optional system message prepended to chat prompts.
        model_name: Label recorded in the eval logs.
    """

    def __init__(
        self,
        model_path: str,
        execution_provider: str = "cuda",
        max_new_tokens: int = 2048,
        extract_letter_choice: bool = False,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 1,
        system_prompt: Optional[str] = None,
        model_name: str = "gpt-oss-20b",
        registry: Any = None,
        **kwargs,
    ):
        if not os.path.isdir(model_path):
            raise ValueError(f"model_path is not a directory: {model_path}")

        self.model_path = model_path
        self.execution_provider = execution_provider
        self.max_new_tokens = int(max_new_tokens)
        self.extract_letter_choice = bool(extract_letter_choice)
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.system_prompt = system_prompt
        self.model_name = model_name

        # Load the chat template shipped alongside the model, if present.
        self.template_str = ""
        jinja_path = os.path.join(model_path, "chat_template.jinja")
        if os.path.exists(jinja_path):
            with open(jinja_path, encoding="utf-8") as f:
                self.template_str = f.read()

        logger.info(
            "Loading ONNX model from %s (provider=%s)", model_path, execution_provider
        )
        config = og.Config(model_path)
        if execution_provider != "follow_config":
            config.clear_providers()
            if execution_provider != "cpu":
                config.append_provider(execution_provider)
        self.model = og.Model(config)
        self.tokenizer = og.Tokenizer(self.model)

        # A single shared model on one GPU cannot run concurrent generations
        # safely, so serialize generation across eval worker threads.
        self._lock = threading.Lock()

    def _build_messages(self, prompt: Union[str, OpenAICreateChatPrompt]) -> list[dict]:
        if is_chat_prompt(prompt):
            messages = [dict(msg) for msg in prompt]
        else:
            messages = [{"role": "user", "content": str(prompt)}]

        if self.system_prompt and not any(m.get("role") == "system" for m in messages):
            messages = [{"role": "system", "content": self.system_prompt}] + messages
        return messages

    def _apply_template(self, messages: list[dict]) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages=json.dumps(messages),
                tools="",
                add_generation_prompt=True,
                template_str=self.template_str,
            )
        except Exception as e:
            logger.warning("apply_chat_template failed (%s); falling back to plain text", e)
            return CompletionPrompt(messages).to_formatted_prompt()

    def __call__(
        self,
        prompt: Union[str, OpenAICreateChatPrompt],
        **kwargs,
    ) -> ORTGenAICompletionResult:
        if isinstance(prompt, Prompt):
            prompt = prompt.to_formatted_prompt()

        messages = self._build_messages(prompt)
        full_prompt = self._apply_template(messages)
        input_tokens = self.tokenizer.encode(full_prompt)
        prompt_len = len(input_tokens)

        search_options: dict[str, Any] = {
            "do_sample": self.do_sample,
            "max_length": prompt_len + self.max_new_tokens,
            "batch_size": 1,
        }
        if self.do_sample:
            search_options["temperature"] = self.temperature
            search_options["top_p"] = self.top_p
            search_options["top_k"] = self.top_k

        with self._lock:
            params = og.GeneratorParams(self.model)
            params.set_search_options(**search_options)
            generator = og.Generator(self.model, params)
            try:
                generator.append_tokens(input_tokens)
                target = generator.token_count() + self.max_new_tokens
                while not generator.is_done() and generator.token_count() < target:
                    generator.generate_next_token()
                sequence = generator.get_sequence(0)
            finally:
                del generator

        new_tokens = sequence[prompt_len:]
        raw = self.tokenizer.decode(new_tokens)
        answer = extract_final_channel(raw)
        if self.extract_letter_choice:
            extracted = extract_choice_letter(answer)
            if extracted:
                answer = extracted

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("RAW completion: %r", raw)
            logger.debug("FINAL answer: %r", answer)

        record_sampling(
            prompt=full_prompt,
            sampled=[answer],
            model=self.model_name,
        )
        return ORTGenAICompletionResult(completion=answer, raw=raw, prompt=full_prompt)
