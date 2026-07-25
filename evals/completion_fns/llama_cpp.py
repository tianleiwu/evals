"""A CompletionFn that serves a local GGUF model with llama.cpp's ``llama-server``.

This mirrors :mod:`evals.completion_fns.ort_genai` so the same evals (GPQA,
MMLU-Pro, AIME, IFEval) can be run against llama.cpp for a side-by-side
comparison with onnxruntime-genai.

Each eval worker process owns one ``llama-server`` subprocess bound to whatever
GPU ``CUDA_VISIBLE_DEVICES`` selects for that worker, so the multi-GPU shard
runner works unchanged. Pass ``server_url`` instead of ``model_path`` to reuse an
already-running server.

For gpt-oss (harmony) the server parses the channels itself: the ``final``
channel arrives as ``message.content`` and the chain-of-thought as
``message.reasoning_content``. The reasoning budget is selected per request via
``chat_template_kwargs={"reasoning_effort": ...}``.
"""
import atexit
import json
import logging
import os
import socket
import subprocess
import threading
import time
from typing import Any, Optional, Union

import requests

from evals.api import CompletionFn, CompletionResult
from evals.completion_fns.ort_genai import (
    extract_choice_letter,
    extract_final_channel,
)
from evals.prompt.base import OpenAICreateChatPrompt, Prompt, is_chat_prompt
from evals.record import record_sampling

logger = logging.getLogger(__name__)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class LlamaCppCompletionResult(CompletionResult):
    def __init__(self, completion: str, raw: str, prompt: Any):
        self.completion = completion
        self.raw = raw
        self.prompt = prompt

    def get_completions(self) -> list[str]:
        return [self.completion]


class LlamaCppCompletionFn(CompletionFn):
    """Generate completions from a local GGUF model via ``llama-server``.

    Args:
        model_path: Path to the ``.gguf`` file (required unless ``server_url``).
        server_binary: Path to the ``llama-server`` executable.
        server_url: Use an already-running server instead of spawning one.
        n_gpu_layers: Layers offloaded to the GPU (99 = all).
        ctx_size: Server context window in tokens.
        flash_attn: Value for ``--flash-attn`` ("on"/"off"/"auto").
        max_new_tokens: Maximum number of tokens generated per sample.
        reasoning_effort: harmony reasoning budget ("low"/"medium"/"high").
        do_sample: If False (default), decode greedily (temperature 0).
        extract_letter_choice/choice_letters: Multiple-choice answer extraction.
        extra_server_args: Extra ``llama-server`` flags (space-separated string).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        server_binary: str = "/home/tianlei/git/llama.cpp/build_cu130/bin/llama-server",
        server_url: Optional[str] = None,
        lib_dirs: str = "",
        n_gpu_layers: int = 99,
        ctx_size: int = 131072,
        flash_attn: str = "on",
        max_new_tokens: int = 2048,
        reasoning_effort: Optional[str] = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 1,
        seed: int = 0,
        extract_letter_choice: bool = False,
        choice_letters: str = "ABCD",
        system_prompt: Optional[str] = None,
        model_name: str = "gpt-oss-20b-llamacpp",
        startup_timeout: float = 600.0,
        request_timeout: float = 7200.0,
        extra_server_args: str = "",
        registry: Any = None,
        **kwargs,
    ):
        self.max_new_tokens = int(max_new_tokens)
        self.reasoning_effort = reasoning_effort
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.seed = int(seed)
        self.extract_letter_choice = bool(extract_letter_choice)
        self.choice_letters = str(choice_letters)
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.request_timeout = float(request_timeout)
        self._process: Optional[subprocess.Popen] = None
        # llama-server handles one request at a time per slot; keep the eval
        # workers serialized like the ort_genai fn does.
        self._lock = threading.Lock()

        if server_url:
            self.server_url = server_url.rstrip("/")
        else:
            if not model_path or not os.path.isfile(model_path):
                raise ValueError(f"model_path is not a gguf file: {model_path}")
            if not os.path.isfile(server_binary) or not os.access(server_binary, os.X_OK):
                raise ValueError(f"server_binary is not executable: {server_binary}")
            port = _free_port()
            self.server_url = f"http://127.0.0.1:{port}"
            self._start_server(
                model_path=model_path,
                server_binary=server_binary,
                lib_dirs=lib_dirs,
                port=port,
                n_gpu_layers=int(n_gpu_layers),
                ctx_size=int(ctx_size),
                flash_attn=flash_attn,
                extra_server_args=extra_server_args,
                startup_timeout=float(startup_timeout),
            )

    # -- server lifecycle ---------------------------------------------------
    def _start_server(
        self,
        model_path: str,
        server_binary: str,
        lib_dirs: str,
        port: int,
        n_gpu_layers: int,
        ctx_size: int,
        flash_attn: str,
        extra_server_args: str,
        startup_timeout: float,
    ) -> None:
        cmd = [
            server_binary,
            "--model", model_path,
            "--host", "127.0.0.1",
            "--port", str(port),
            "--n-gpu-layers", str(n_gpu_layers),
            "--ctx-size", str(ctx_size),
            "--flash-attn", flash_attn,
            "--parallel", "1",
            "--jinja",
            "--no-warmup",
        ]
        if extra_server_args:
            cmd += extra_server_args.split()

        env = dict(os.environ)
        search_dirs = [d for d in lib_dirs.split(":") if d]
        search_dirs.append(os.path.dirname(server_binary))
        cuda_home = env.get("CUDA_HOME", "")
        if cuda_home:
            search_dirs.append(os.path.join(cuda_home, "lib64"))
        env["LD_LIBRARY_PATH"] = ":".join(search_dirs + [env.get("LD_LIBRARY_PATH", "")])

        log_path = os.path.join(
            os.environ.get("LLAMACPP_LOG_DIR", "/tmp"),
            f"llama_server_{port}.log",
        )
        logger.info("starting llama-server on port %s (log: %s)", port, log_path)
        # ruff: the log file is intentionally kept open for the server's lifetime.
        self._log_file = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
        self._process = subprocess.Popen(
            cmd, stdout=self._log_file, stderr=subprocess.STDOUT, env=env
        )
        atexit.register(self.close)

        deadline = time.time() + startup_timeout
        while time.time() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited with code {self._process.returncode}; see {log_path}"
                )
            try:
                resp = requests.get(f"{self.server_url}/health", timeout=5)
                if resp.status_code == 200:
                    logger.info("llama-server ready on port %s", port)
                    return
            except requests.RequestException:
                pass
            time.sleep(2)
        self.close()
        raise RuntimeError(f"llama-server did not become ready within {startup_timeout}s; see {log_path}")

    def close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
        log_file = getattr(self, "_log_file", None)
        if log_file is not None:
            log_file.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # -- generation ---------------------------------------------------------
    def _build_messages(self, prompt: Union[str, OpenAICreateChatPrompt]) -> list[dict]:
        if is_chat_prompt(prompt):
            messages = [dict(msg) for msg in prompt]
        else:
            messages = [{"role": "user", "content": str(prompt)}]
        if self.system_prompt and not any(m.get("role") == "system" for m in messages):
            messages = [{"role": "system", "content": self.system_prompt}] + messages
        return messages

    def __call__(
        self,
        prompt: Union[str, OpenAICreateChatPrompt],
        **kwargs,
    ) -> LlamaCppCompletionResult:
        if isinstance(prompt, Prompt):
            prompt = prompt.to_formatted_prompt()
        messages = self._build_messages(prompt)

        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "stream": False,
            "cache_prompt": True,
            "seed": self.seed,
        }
        if self.do_sample:
            payload["temperature"] = self.temperature
            payload["top_p"] = self.top_p
            payload["top_k"] = self.top_k
        else:
            payload["temperature"] = 0.0
            payload["top_k"] = 1
        if self.reasoning_effort:
            payload["chat_template_kwargs"] = {"reasoning_effort": self.reasoning_effort}

        with self._lock:
            response = requests.post(
                f"{self.server_url}/v1/chat/completions",
                json=payload,
                timeout=self.request_timeout,
            )
        response.raise_for_status()
        body = response.json()
        message = body["choices"][0]["message"]
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""

        # The server normally strips the harmony markers; run the same extractor
        # as the ort_genai fn so a raw passthrough is still handled.
        answer = extract_final_channel(content) if content else extract_final_channel(reasoning)
        if self.extract_letter_choice:
            extracted = extract_choice_letter(answer, self.choice_letters)
            if extracted:
                answer = extracted

        raw = json.dumps(
            {"reasoning_content": reasoning, "content": content}, ensure_ascii=False
        )
        record_sampling(
            prompt=json.dumps(messages, ensure_ascii=False),
            sampled=[answer],
            model=self.model_name,
        )
        return LlamaCppCompletionResult(completion=answer, raw=raw, prompt=messages)
