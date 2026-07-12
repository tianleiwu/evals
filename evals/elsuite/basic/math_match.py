"""MathMatch: grade free-form math answers by symbolic equivalence.

Unlike :class:`~evals.elsuite.basic.match.Match` (which does a letter/prefix
string comparison), this eval parses the model's final answer -- preferring a
``\\boxed{...}`` expression -- and the gold answer, then checks *mathematical*
equivalence with ``math_verify`` (sympy-backed). It is used for the MATH
benchmark, whose answers are LaTeX expressions (fractions, intervals, matrices,
...) that are not comparable as plain strings.

It records the same ``match`` events as ``Match`` (``data.correct`` boolean), so
the parallel-eval summary/report tooling grades it identically.
"""

from typing import Any

import logging

import evals
import evals.metrics
from evals.api import CompletionFn
from evals.record import record_match

# math_verify logs a warning on every call when its signal-based timeout is
# disabled (which we do below so it works in the harness's worker threads).
# Silence it to avoid one line of noise per graded sample.
logging.getLogger("math_verify").setLevel(logging.ERROR)
logging.getLogger("math_verify.utils").setLevel(logging.ERROR)


def _math_equal(gold: str, sampled: str) -> bool:
    """Return True if ``sampled`` is mathematically equivalent to ``gold``.

    ``gold`` is the reference answer (bare LaTeX, e.g. ``\\frac{1}{2}``);
    ``sampled`` is the raw model text, from which ``math_verify`` extracts the
    ``\\boxed{...}`` (or last) expression.
    """
    from math_verify import parse, verify  # lazy: heavy optional dependency

    gold = (gold or "").strip()
    sampled = (sampled or "").strip()
    if not gold or not sampled:
        return False

    # math_verify's default timeout uses signal.alarm(), which only works on the
    # main thread and raises inside eval worker threads. Disable it (MATH answers
    # parse quickly) so grading works under the eval harness's threaded runner.
    def _parse(text):
        try:
            return parse(text, parsing_timeout=None)
        except Exception:  # noqa: BLE001 - malformed LaTeX -> treat as unparseable
            return None

    # Parse the gold answer both bare and $-wrapped so bare LaTeX still parses.
    gold_forms = []
    for candidate in (f"${gold}$", gold):
        parsed = _parse(candidate)
        if parsed:
            gold_forms.append(parsed)
    if not gold_forms:
        return False

    pred = _parse(sampled)
    if not pred:
        return False

    for gold_parsed in gold_forms:
        try:
            if verify(gold_parsed, pred, timeout_seconds=None) or verify(
                pred, gold_parsed, timeout_seconds=None
            ):
                return True
        except Exception:  # noqa: BLE001 - verification can raise on odd inputs
            continue
    return False


class MathMatch(evals.Eval):
    """Grade MATH-style answers by symbolic equivalence (math_verify)."""

    def __init__(
        self,
        completion_fns: list[CompletionFn],
        samples_jsonl: str,
        *args,
        max_tokens: int = 2048,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "MathMatch only supports one completion fn"
        self.max_tokens = max_tokens
        self.samples_jsonl = samples_jsonl
        try:
            import math_verify  # noqa: F401, PLC0415
        except ImportError as e:
            raise ImportError(
                "MathMatch requires math_verify. Install with: pip install math_verify"
            ) from e

    def eval_sample(self, sample: Any, *_):
        assert isinstance(sample, dict), "sample must be a dict"
        assert "input" in sample, "sample must have an 'input' key"
        assert "ideal" in sample, "sample must have an 'ideal' key"

        result = self.completion_fn(prompt=sample["input"], temperature=0.0)
        sampled = result.get_completions()[0]

        gold = sample["ideal"]
        if isinstance(gold, list):
            correct = any(_math_equal(g, sampled) for g in gold)
        else:
            correct = _math_equal(gold, sampled)

        # Keep only a short prefix of the (potentially long) answer in the record.
        record_match(bool(correct), expected=gold, picked=sampled[:256])
        return correct

    def run(self, recorder):
        samples = self.get_samples()
        self.eval_all_samples(recorder, samples)
        events = recorder.get_events("match")
        return {
            "accuracy": evals.metrics.get_accuracy(events),
            "boostrap_std": evals.metrics.get_bootstrap_accuracy_std(events),
        }
