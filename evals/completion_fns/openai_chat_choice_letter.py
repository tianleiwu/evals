import re
from typing import Any, Optional, Union

from openai import OpenAI

from evals.base import CompletionFnSpec
from evals.completion_fns.openai import (
    OpenAIBaseCompletionResult,
    openai_chat_completion_create_retrying,
)
from evals.prompt.base import ChatCompletionPrompt, OpenAICreateChatPrompt, Prompt
from evals.record import record_sampling


_ANSWER_PATTERNS = [
    re.compile(r"^\s*([ABCD])\s*[\.)]?\s*$", re.IGNORECASE),
    re.compile(r"\banswer\s*[:\-]?\s*([ABCD])\b", re.IGNORECASE),
    re.compile(r"\bcorrect answer(?:\s+is|\s*:)\s*([ABCD])\b", re.IGNORECASE),
    re.compile(r"\boption\s*([ABCD])\b", re.IGNORECASE),
]


class OpenAIChatChoiceLetterResult(OpenAIBaseCompletionResult):
    @staticmethod
    def _extract_choice(message: Any) -> str:
        content = (getattr(message, "content", None) or "").strip()
        reasoning = (getattr(message, "reasoning_content", None) or "").strip()

        # Prefer the final visible assistant message when present.
        for text in [content, reasoning]:
            if not text:
                continue
            for pattern in _ANSWER_PATTERNS:
                m = pattern.search(text)
                if m:
                    return m.group(1).upper()

        # Fall back to first standalone letter only if content itself is short.
        if content and len(content) <= 8:
            m = re.search(r"\b([ABCD])\b", content, re.IGNORECASE)
            if m:
                return m.group(1).upper()

        return ""

    def get_completions(self) -> list[str]:
        completions: list[str] = []
        if self.raw_data:
            for choice in self.raw_data.choices:
                completions.append(self._extract_choice(choice.message))
        return completions


class OpenAIChatChoiceLetterFn(CompletionFnSpec):
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
    ) -> OpenAIChatChoiceLetterResult:
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
        result = OpenAIChatChoiceLetterResult(raw_data=result, prompt=openai_create_prompt)
        record_sampling(
            prompt=result.prompt,
            sampled=result.get_completions(),
            model=result.raw_data.model,
            usage=result.raw_data.usage,
        )
        return result
