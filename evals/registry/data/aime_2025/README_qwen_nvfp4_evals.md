# Qwen3.6-35B-A3B-NVFP4 extra evals (added for the NVIDIA metric table)

These reproduce parts of the NVIDIA NVFP4 eval table using the ORT-GenAI
completion functions in `completion_fns/ort_genai.yaml`. All run through the
sharded H200 runners under `dev/scripts/h200_18/`.

## Added and working

### AIME 2025  (`aime_2025`, class `MathMatch`)
- Data: `data/aime_2025/samples.jsonl` (30 problems from `math-ai/aime25`).
- Completion fn: `qwen/qwen3.6-nvfp4-aime` (thinking on, `\boxed{}` grading).
- Run: `bash dev/scripts/h200_18/run_aime_qwen_nvfp4_sharded.sh`
- NOTE: AIME reasoning traces are very long. With `max_new_tokens=32768` many
  solutions truncate before the final `\boxed{}` (observed 15/30 = 50%). The
  NVIDIA reference uses a 131072-token budget (~24 min/problem at ~90 tok/s).
  Raise `max_new_tokens` on the completion fn for a comparable number.

### IFEval  (`ifeval`, class `IFEval`)  -- proxy for the reported IFBench
- Verifiers ported verbatim from lm-evaluation-harness / google-research into
  `evals/elsuite/ifeval/` (`instructions*.py` + the `IFEval` eval class).
- Data: `data/ifeval/samples.jsonl` (541 prompts from `google/IFEval`).
- Completion fn: `qwen/qwen3.6-nvfp4-ifeval` (NON-thinking; the verifiers grade
  the literal response, so `disable_thinking: true`).
- Run: `bash dev/scripts/h200_18/run_ifeval_qwen_nvfp4_sharded.sh`
- Reports the 4 IFEval metrics; the sharded headline `accuracy` is prompt-level
  strict. Deps: `langdetect immutabledict "nltk>=3.9.1"` (+ `punkt_tab`).
- NOTE: the NVIDIA table lists **IFBench** (allenai/IFBench), a distinct, newer
  benchmark NOT in lm-eval. IFEval here is the closest ready reference.

## Not added (blocked)

### SciCode
Needs a code-execution harness: generate Python per scientific subproblem and
run numerical unit tests against gold solutions, with scientific deps and a
sandbox. Substantial infra; not built here.

### MMMU-PRO
Multimodal (image inputs). The exported model is **text-only** (`text.onnx`,
`qwen3_5_moe_text`), so MMMU-PRO cannot run against it. Requires a vision ONNX
export plus a multimodal genai path.
