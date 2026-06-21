import re
from typing import Any, Optional, Union

from openai import OpenAI

from evals.api import CompletionResult
from evals.base import CompletionFnSpec
from evals.completion_fns.openai import openai_chat_completion_create_retrying
from evals.prompt.base import ChatCompletionPrompt, OpenAICreateChatPrompt, Prompt
from evals.record import record_sampling


_LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
_BRACKET_LETTER_RE = re.compile(r"\[\s*([ABCD])\s*\]", re.IGNORECASE)
_ANSWER_MARKER_RE = re.compile(
    r"(?:^|\n|\r)\s*(?:final\s+answer|answer|correct\s+answer)\s*[:\-]?\s*([ABCD])\b",
    re.IGNORECASE,
)


def _extract_letter(text: str) -> str:
    if not text:
        return ""
    # Prefer explicit final-answer markers when present.
    marker = _ANSWER_MARKER_RE.findall(text)
    if marker:
        return marker[-1].upper()

    # Next prefer bracketed option format like [C].
    bracket = _BRACKET_LETTER_RE.findall(text)
    if bracket:
        return bracket[-1].upper()

    # Fallback: take the last standalone A/B/C/D token.
    tokens = _LETTER_RE.findall(text)
    return tokens[-1].upper() if tokens else ""


def _extract_from_choice(choice: Any) -> str:
    message = getattr(choice, "message", None)
    content = getattr(message, "content", "") if message is not None else ""
    letter = _extract_letter(content or "")
    if letter:
        return letter

    # Some backends return reasoning text separately and keep content empty.
    reasoning = getattr(message, "reasoning_content", "") if message is not None else ""
    letter = _extract_letter(reasoning or "")
    if letter:
        return letter

    text = getattr(choice, "text", "")
    return _extract_letter(text or "")


class OpenAICompletionLetterResult(CompletionResult):
    def __init__(self, raw_data: Any, prompt: Any):
        self.raw_data = raw_data
        self.prompt = prompt

    def get_completions(self) -> list[str]:
        completions: list[str] = []
        if self.raw_data:
            for choice in self.raw_data.choices:
                completions.append(_extract_from_choice(choice))
        return completions


class OpenAIChatToCompletionLetterFn(CompletionFnSpec):
    def __init__(
        self,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        n_ctx: Optional[int] = None,
        extra_options: Optional[dict] = {},
        registry: Any = None,
        **kwargs,
    ):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.n_ctx = n_ctx
        self.extra_options = extra_options
        _ = registry
        _ = kwargs

    def __call__(
        self,
        prompt: Union[str, OpenAICreateChatPrompt],
        **kwargs,
    ) -> OpenAICompletionLetterResult:
        if not isinstance(prompt, Prompt):
            assert (
                isinstance(prompt, str)
                or (isinstance(prompt, list) and all(isinstance(token, int) for token in prompt))
                or (isinstance(prompt, list) and all(isinstance(token, str) for token in prompt))
                or (isinstance(prompt, list) and all(isinstance(msg, dict) for msg in prompt))
            ), f"Got type {type(prompt)}, with val {type(prompt[0])} for prompt, expected str or list[int] or list[str] or list[dict[str, str]]"

            prompt = ChatCompletionPrompt(raw_prompt=prompt)

        openai_create_prompt: OpenAICreateChatPrompt = prompt.to_formatted_prompt()

        result = openai_chat_completion_create_retrying(
            OpenAI(api_key=self.api_key, base_url=self.api_base),
            model=self.model,
            messages=openai_create_prompt,
            **{**kwargs, **self.extra_options},
        )
        result = OpenAICompletionLetterResult(raw_data=result, prompt=openai_create_prompt)
        record_sampling(
            prompt=result.prompt,
            sampled=result.get_completions(),
            model=result.raw_data.model,
            usage=result.raw_data.usage,
        )
        return result
