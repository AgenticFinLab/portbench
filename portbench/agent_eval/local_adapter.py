"""
Local model adapter for PortBench evaluation without external API calls.

Supports three local serving backends:
  - vLLM          (high-throughput OpenAI-compatible server)
  - Ollama        (desktop-friendly model runner)
  - HuggingFace   (direct transformers inference, no server required)

All three implement the same AgentAdapter interface as the cloud adapters,
so they can be dropped into build_default_pipeline() with zero other changes.

Usage:
    # vLLM server (start with: python -m vllm.entrypoints.openai.api_server --model ...)
    from portbench.agent_eval.local_adapter import VLLMAdapter
    adapter = VLLMAdapter(model="meta-llama/Llama-3.1-8B-Instruct", base_url="http://localhost:8000")

    # Ollama (start with: ollama serve; ollama pull llama3.1)
    from portbench.agent_eval.local_adapter import OllamaAdapter
    adapter = OllamaAdapter(model="llama3.1")

    # HuggingFace (loads model locally; slow first call, cached afterwards)
    from portbench.agent_eval.local_adapter import HuggingFaceAdapter
    adapter = HuggingFaceAdapter(model_name="microsoft/Phi-3-mini-4k-instruct")

    # Then use identically to cloud adapters:
    pipeline = build_default_pipeline(adapter)
    result = pipeline.run_episode(snapshot)
"""

import time
import torch

from .base import AgentAdapter


# ---------------------------------------------------------------------------
# Shared retry helper (same interface as llm_adapters.py)
# ---------------------------------------------------------------------------


def _retry(fn, max_retries: int = 3, base_delay: float = 1.0):
    """Retry fn() with exponential backoff. Skips non-retryable connection errors."""
    delay = base_delay
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if attempt < max_retries - 1:
                print(
                    f"  Local model error (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {delay:.0f}s…"
                )
                time.sleep(delay)
                delay *= 2
            else:
                raise


# ---------------------------------------------------------------------------
# vLLM Adapter
# ---------------------------------------------------------------------------


class VLLMAdapter(AgentAdapter):
    """
    AgentAdapter for a locally running vLLM OpenAI-compatible server.

    vLLM exposes the same HTTP API as OpenAI, so this adapter uses the
    openai SDK pointed at a local base_url.

    Prerequisites:
        pip install vllm openai
        python -m vllm.entrypoints.openai.api_server \\
            --model meta-llama/Llama-3.1-8B-Instruct \\
            --port 8000

    Args:
        model:       Model name as registered in the vLLM server (must match --model flag).
        base_url:    URL of the vLLM OpenAI-compatible endpoint (default: http://localhost:8000/v1).
        max_tokens:  Maximum tokens to generate.
        temperature: Sampling temperature (0.0 = greedy).
        system_prompt: System message prepended to every call.
        max_retries:   Retry attempts on connection errors.
        timeout:       HTTP timeout in seconds.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        max_tokens: int = 2048,
        temperature: float = 0.0,
        system_prompt: str = "You are a professional portfolio manager. "
        "Respond with structured JSON as instructed.",
        max_retries: int = 3,
        timeout: float = 120.0,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            )

        self._model = model
        self._base_url = base_url.rstrip("/")
        self._client = OpenAI(
            base_url=self._base_url,
            api_key="EMPTY",  # vLLM does not require a real API key
            timeout=timeout,
        )
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._system_prompt = system_prompt
        self._max_retries = max_retries

    @property
    def model_name(self) -> str:
        return f"vllm/{self._model}"

    def complete(self, prompt: str) -> str:
        """Send prompt to vLLM and return the response text."""

        def _call():
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            return response.choices[0].message.content

        return _retry(_call, max_retries=self._max_retries)


# ---------------------------------------------------------------------------
# Ollama Adapter
# ---------------------------------------------------------------------------


class OllamaAdapter(AgentAdapter):
    """
    AgentAdapter for Ollama — a local model runner for open-source LLMs.

    Ollama exposes a simple HTTP API. Models are pulled once and cached locally.

    Prerequisites:
        # Install Ollama: https://ollama.com/download
        ollama serve          # Start Ollama server (or it auto-starts on macOS)
        ollama pull llama3.1  # Pull the model (one-time download)

    Then in Python:
        adapter = OllamaAdapter(model="llama3.1")

    Args:
        model:       Ollama model tag (e.g., "llama3.1", "mistral", "phi3:mini").
        host:        Ollama server URL (default: http://localhost:11434).
        max_tokens:  Maximum tokens to generate (mapped to num_predict).
        temperature: Sampling temperature.
        system_prompt: System message.
        max_retries:   Retry attempts on connection errors.
        timeout:       HTTP timeout in seconds.

    For a full list of available models, see https://ollama.com/library
    """

    def __init__(
        self,
        model: str = "llama3.1",
        host: str = "http://localhost:11434",
        max_tokens: int = 2048,
        temperature: float = 0.0,
        system_prompt: str = "You are a professional portfolio manager. "
        "Respond with structured JSON as instructed.",
        max_retries: int = 3,
        timeout: float = 120.0,
    ):
        self._model = model
        self._host = host.rstrip("/")
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._system_prompt = system_prompt
        self._max_retries = max_retries
        self._timeout = timeout

    @property
    def model_name(self) -> str:
        return f"ollama/{self._model}"

    def complete(self, prompt: str) -> str:
        """Send prompt to Ollama and return the response text."""
        try:
            import requests
        except ImportError:
            raise ImportError(
                "requests package required. Install with: pip install requests"
            )

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "num_predict": self._max_tokens,
                "temperature": self._temperature,
            },
        }

        def _call():
            resp = requests.post(
                f"{self._host}/api/chat",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"]

        return _retry(_call, max_retries=self._max_retries)


# ---------------------------------------------------------------------------
# HuggingFace Adapter
# ---------------------------------------------------------------------------


class HuggingFaceAdapter(AgentAdapter):
    """
    AgentAdapter for direct HuggingFace transformers inference (no server required).

    Loads the model into local GPU/CPU memory using the transformers library.
    The model is loaded once on first instantiation and reused for all calls.

    Prerequisites:
        pip install transformers accelerate torch

    Args:
        model_name:       HuggingFace model ID (e.g., "microsoft/Phi-3-mini-4k-instruct").
        device:           "cuda", "mps", "cpu", or "auto" (default: "auto").
        max_new_tokens:   Maximum tokens to generate per response.
        temperature:      Sampling temperature (set to 0 for greedy decoding).
        do_sample:        Whether to sample from the distribution (False = greedy).
        system_prompt:    System message prepended to the conversation.
        torch_dtype:      Data type for model weights ("auto", "float16", "bfloat16").
        load_in_4bit:     Load the model in 4-bit quantization (requires bitsandbytes).
        trust_remote_code: Allow running custom model code from the Hub (needed for some models).
        max_retries:      Retry attempts on generation errors.

    Recommended models for CPU/limited GPU:
        "microsoft/Phi-3-mini-4k-instruct"     (3.8B, fast on CPU)
        "mistralai/Mistral-7B-Instruct-v0.3"   (7B, good quality)
        "meta-llama/Meta-Llama-3.1-8B-Instruct" (8B, needs 16GB VRAM or 4-bit quant)

    Note:
        For high-throughput batch evaluation, prefer VLLMAdapter over this adapter.
        HuggingFaceAdapter is best for one-off experiments and model exploration.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        do_sample: bool = False,
        system_prompt: str = "You are a professional portfolio manager. "
        "Respond with structured JSON as instructed.",
        torch_dtype: str = "auto",
        load_in_4bit: bool = False,
        trust_remote_code: bool = False,
        max_retries: int = 2,
    ):
        try:
            import transformers
        except ImportError:
            raise ImportError(
                "transformers package required. Install with: pip install transformers accelerate"
            )

        self._model_name = model_name
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._do_sample = do_sample or (temperature > 0)
        self._system_prompt = system_prompt
        self._max_retries = max_retries

        print(f"Loading {model_name} (device={device}, 4bit={load_in_4bit})…")

        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        resolved_dtype = dtype_map.get(torch_dtype, "auto")

        load_kwargs = dict(
            pretrained_model_name_or_path=model_name,
            device_map=device,
            torch_dtype=resolved_dtype,
            trust_remote_code=trust_remote_code,
        )

        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            except ImportError:
                raise ImportError(
                    "bitsandbytes required for 4-bit quantization. "
                    "Install with: pip install bitsandbytes"
                )

        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self._model = transformers.AutoModelForCausalLM.from_pretrained(**load_kwargs)
        self._pipeline = transformers.pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
        )

        print(f"  Model loaded successfully.")

    @property
    def model_name(self) -> str:
        return f"hf/{self._model_name}"

    def complete(self, prompt: str) -> str:
        """Run inference using the HuggingFace pipeline and return the generated text."""
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt},
        ]

        gen_kwargs = dict(
            max_new_tokens=self._max_new_tokens,
            do_sample=self._do_sample,
            pad_token_id=self._tokenizer.eos_token_id,
            return_full_text=False,
        )
        if self._do_sample and self._temperature > 0:
            gen_kwargs["temperature"] = self._temperature

        def _call():
            output = self._pipeline(messages, **gen_kwargs)
            # Pipeline returns a list of dicts; extract the generated content
            if isinstance(output, list) and len(output) > 0:
                first = output[0]
                if isinstance(first, dict) and "generated_text" in first:
                    generated = first["generated_text"]
                    # generated_text may be a list of messages or a plain string
                    if isinstance(generated, list):
                        # Extract the last assistant message
                        for msg in reversed(generated):
                            if isinstance(msg, dict) and msg.get("role") == "assistant":
                                return msg["content"]
                        return str(generated[-1])
                    return str(generated)
            return str(output)

        return _retry(_call, max_retries=self._max_retries)
