"""IFEval: Google's Instruction-Following Eval, ported into the evals harness.

Each sample carries a natural-language ``prompt`` plus a list of programmatic
instruction ids (``instruction_id_list``) and their ``kwargs``. The model's raw
response is checked against every instruction's verifier (ported verbatim from
lm-evaluation-harness / the original google-research IFEval). We report the four
standard IFEval metrics:

- ``prompt_level_strict_acc``  -- all instructions followed (exact response)
- ``inst_level_strict_acc``    -- fraction of individual instructions followed
- ``prompt_level_loose_acc``   -- all followed under lenient response variants
- ``inst_level_loose_acc``     -- fraction followed under lenient variants

The headline ``accuracy`` recorded via ``match`` events equals
``prompt_level_strict_acc`` so the shared parallel-eval summary tooling grades it
identically to the other benchmarks.
"""

import dataclasses
import threading
from typing import Any, Dict, List, Optional, Union

import evals
import evals.metrics
from evals.api import CompletionFn
from evals.record import record_match

from evals.elsuite.ifeval import instructions_registry


@dataclasses.dataclass
class _InputExample:
    key: int
    instruction_id_list: List[str]
    prompt: str
    kwargs: List[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class _OutputExample:
    follow_all_instructions: bool
    follow_instruction_list: List[bool]


def _test_following(inp: _InputExample, response: str, responses: List[str]) -> _OutputExample:
    """Check ``responses`` against every instruction. ``responses`` is a list of
    candidate strings; an instruction counts as followed if ANY candidate follows
    it (strict passes a single-element list; loose passes lenient variants)."""
    is_following_list = []
    for index, instruction_id in enumerate(inp.instruction_id_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        kwargs = {k: v for k, v in inp.kwargs[index].items() if v}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        is_following = False
        for r in responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break
        is_following_list.append(is_following)

    return _OutputExample(
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def _loose_variants(response: str) -> List[str]:
    r = response.split("\n")
    remove_first = "\n".join(r[1:]).strip()
    remove_last = "\n".join(r[:-1]).strip()
    remove_both = "\n".join(r[1:-1]).strip()
    return [
        response,
        response.replace("*", ""),
        remove_first,
        remove_last,
        remove_both,
        remove_first.replace("*", ""),
        remove_last.replace("*", ""),
        remove_both.replace("*", ""),
    ]


class IFEval(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        samples_jsonl: str,
        *args,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "IFEval only supports one completion fn"
        self.samples_jsonl = samples_jsonl
        self._lock = threading.Lock()
        self._records: List[Dict[str, Any]] = []

    def eval_sample(self, sample: Any, *_):
        assert isinstance(sample, dict), "sample must be a dict"
        assert "input" in sample, "sample must have an 'input' key"
        assert "instruction_id_list" in sample, "sample must have 'instruction_id_list'"
        assert "kwargs" in sample, "sample must have 'kwargs'"

        result = self.completion_fn(prompt=sample["input"], temperature=0.0)
        response = result.get_completions()[0]

        inp = _InputExample(
            key=sample.get("key", 0),
            instruction_id_list=sample["instruction_id_list"],
            prompt=sample.get("prompt", ""),
            kwargs=sample["kwargs"],
        )

        strict = _test_following(inp, response, [response])
        loose = _test_following(inp, response, _loose_variants(response))

        with self._lock:
            self._records.append(
                {
                    "prompt_strict": strict.follow_all_instructions,
                    "inst_strict": strict.follow_instruction_list,
                    "prompt_loose": loose.follow_all_instructions,
                    "inst_loose": loose.follow_instruction_list,
                }
            )

        # Headline accuracy = prompt-level strict.
        record_match(
            bool(strict.follow_all_instructions),
            expected="follow_all_instructions",
            picked=response[:256],
        )
        return strict.follow_all_instructions

    def run(self, recorder):
        samples = self.get_samples()
        self.eval_all_samples(recorder, samples)
        events = recorder.get_events("match")

        def _inst_acc(key):
            flat = [b for rec in self._records for b in rec[key]]
            return sum(flat) / len(flat) if flat else 0.0

        n = len(self._records) or 1
        return {
            "accuracy": evals.metrics.get_accuracy(events),
            "prompt_level_strict_acc": sum(r["prompt_strict"] for r in self._records) / n,
            "inst_level_strict_acc": _inst_acc("inst_strict"),
            "prompt_level_loose_acc": sum(r["prompt_loose"] for r in self._records) / n,
            "inst_level_loose_acc": _inst_acc("inst_loose"),
        }
