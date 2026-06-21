from typing import Any

from evals.completion_fns.openai import OpenAIChatCompletionFn


class OpenAIChatCompletionFnCompat(OpenAIChatCompletionFn):
    """Compatibility wrapper for registry-injected kwargs.

    evals.registry.make_completion_fn injects a `registry` kwarg into every
    completion function spec. The stock OpenAIChatCompletionFn does not accept
    arbitrary kwargs, so we drop unsupported keys here.
    """

    def __init__(self, registry: Any = None, **kwargs: Any):
        _ = registry
        super().__init__(**kwargs)
