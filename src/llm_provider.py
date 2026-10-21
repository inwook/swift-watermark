import os
import re
import time
import threading
from abc import ABC, abstractmethod

from dotenv import load_dotenv

# Keys come from the environment or a repo-root .env (see .env.example), and shell variables take precedence.
# get_provider() raises if a needed key is unset.
load_dotenv()
openai_api_key    = os.environ.get("OPENAI_API_KEY")
gemini_api_key    = os.environ.get("GEMINI_API_KEY")
hf_api_key        = os.environ.get("HF_API_KEY")
anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

_ENV_VAR_FOR_PROVIDER = {
    "OpenAIProvider":      "OPENAI_API_KEY",
    "GeminiProvider":      "GEMINI_API_KEY",
    "AnthropicProvider":   "ANTHROPIC_API_KEY",
    "HuggingFaceProvider": "HF_API_KEY",
}


# Process-wide token counters shared by all providers. Callers diff get_usage() before and after.
_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "n_calls": 0}
_USAGE_LOCK = threading.Lock()


def _add_usage(prompt_tokens: int, completion_tokens: int) -> None:
    with _USAGE_LOCK:
        _USAGE["prompt_tokens"] += prompt_tokens
        _USAGE["completion_tokens"] += completion_tokens
        _USAGE["total_tokens"] += prompt_tokens + completion_tokens
        _USAGE["n_calls"] += 1


def get_usage() -> dict:
    with _USAGE_LOCK:
        return dict(_USAGE)


def reset_usage() -> None:
    with _USAGE_LOCK:
        for k in _USAGE:
            _USAGE[k] = 0


def diff_usage(before: dict, after: dict) -> dict:
    return {k: after.get(k, 0) - before.get(k, 0) for k in after}


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        pass


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        retries = 5
        for i in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=1.0,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    _add_usage(getattr(usage, "prompt_tokens", 0) or 0,
                               getattr(usage, "completion_tokens", 0) or 0)
                return resp.choices[0].message.content or ""
            except Exception as e:
                if i == retries - 1:
                    print(f"[!] OpenAI API Error after {retries} attempts: {e}")
                    return ""
                time.sleep(2 ** i)
        return ""


class GeminiProvider(LLMProvider):
    def __init__(self, model: str, api_key: str):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        from google.genai import types
        retries = 5
        for i in range(retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.0  # 0 for consistency
                    ),
                    contents=user_prompt
                )
                meta = getattr(response, "usage_metadata", None)
                if meta is not None:
                    _add_usage(getattr(meta, "prompt_token_count", 0) or 0,
                               getattr(meta, "candidates_token_count", 0) or 0)
                return response.text
            except Exception as e:
                if i == retries - 1:
                    print(f"[!] Gemini API Error after {retries} attempts: {e}")
                    return ""
                time.sleep(2 ** i)
        return ""


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str, api_key: str):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        retries = 5
        for i in range(retries):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    _add_usage(getattr(usage, "input_tokens", 0) or 0,
                               getattr(usage, "output_tokens", 0) or 0)
                blocks = resp.content or []
                if blocks and hasattr(blocks[0], "text"):
                    return blocks[0].text or ""
                return ""
            except Exception as e:
                if i == retries - 1:
                    print(f"[!] Anthropic API Error after {retries} attempts: {e}")
                    return ""
                time.sleep(2 ** i)
        return ""


class HuggingFaceProvider(LLMProvider):
    def __init__(self, model: str, api_key: str):
        from huggingface_hub import InferenceClient
        self.client = InferenceClient(api_key=api_key)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        retries = 5
        for i in range(retries):
            try:
                response = self.client.chat_completion(
                    model=self.model,
                    messages=messages,
                    max_tokens=4096,
                    temperature=1.0
                )
                usage = getattr(response, "usage", None)
                if usage is not None:
                    _add_usage(getattr(usage, "prompt_tokens", 0) or 0,
                               getattr(usage, "completion_tokens", 0) or 0)
                return response.choices[0].message.content or ""
            except Exception as e:
                if i == retries - 1:
                    print(f"[!] Hugging Face API Error after {retries} attempts: {e}")
                    return ""
                time.sleep(2 ** i)
        return ""


# model_name -> (provider_class, api_key). Unlisted names fall back to the prefix routing in get_provider().
_MODEL_REGISTRY = {
    "gpt-4o-mini":     (OpenAIProvider, openai_api_key),
    "gpt-4o":          (OpenAIProvider, openai_api_key),
    "gpt-4.1":         (OpenAIProvider, openai_api_key),
    "gpt-4.1-mini":    (OpenAIProvider, openai_api_key),
    "gemini-2.5-flash": (GeminiProvider, gemini_api_key),
    "gemini-2.5-pro":   (GeminiProvider, gemini_api_key),
    "claude-haiku-4-5":            (AnthropicProvider, anthropic_api_key),
    "claude-haiku-4-5-20251001":   (AnthropicProvider, anthropic_api_key),
    "claude-sonnet-4-6":           (AnthropicProvider, anthropic_api_key),
    "claude-opus-4-7":             (AnthropicProvider, anthropic_api_key),
    "meta-llama/Meta-Llama-3-8B-Instruct": (HuggingFaceProvider, hf_api_key),
    "Qwen/Qwen2.5-7B-Instruct":             (HuggingFaceProvider, hf_api_key),
}

DEFAULT_MODEL = "gpt-4o-mini"


def resolve_model(model: str | None = None) -> str:
    """Return the canonical model name to use (DEFAULT_MODEL if None/empty)."""
    return model or DEFAULT_MODEL


def get_provider(model: str | None = None) -> LLMProvider:
    """Build an LLMProvider for model from _MODEL_REGISTRY, else by name prefix."""
    model = resolve_model(model)
    if model in _MODEL_REGISTRY:
        cls, key = _MODEL_REGISTRY[model]
    elif model.startswith(("gpt-", "o1", "o3", "o4")):
        cls, key = OpenAIProvider, openai_api_key
    elif model.startswith("gemini-"):
        cls, key = GeminiProvider, gemini_api_key
    elif model.startswith("claude-"):
        cls, key = AnthropicProvider, anthropic_api_key
    elif "/" in model:
        cls, key = HuggingFaceProvider, hf_api_key
    else:
        raise ValueError(
            f"Unknown LLM model '{model}'. Add it to llm_provider._MODEL_REGISTRY "
            f"or use a recognized prefix (gpt-, gemini-, claude-, <org>/<name>)."
        )
    if not key:
        env_var = _ENV_VAR_FOR_PROVIDER.get(cls.__name__, "<provider env var>")
        raise RuntimeError(
            f"API key for {cls.__name__} is not set. "
            f"Export {env_var} (e.g. via a .env file) before using model '{model}'."
        )
    return cls(model=model, api_key=key)


def model_tag(model: str | None = None) -> str:
    """Filename-safe tag for a model name."""
    name = resolve_model(model)
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")


def _test_provider_usage(label: str, provider: LLMProvider,
                          sys_prompt: str, usr_prompt: str) -> None:
    before = get_usage()
    resp = provider.generate(sys_prompt, usr_prompt)
    after = get_usage()
    d = diff_usage(before, after)
    print(f"\n--- {label} ---")
    print(f"resp: {resp!r}")
    print(f"usage: calls={d['n_calls']}  prompt={d['prompt_tokens']}  "
          f"completion={d['completion_tokens']}  total={d['total_tokens']}")
    assert d["n_calls"] == 1, f"{label}: n_calls expected 1, got {d['n_calls']}"
    assert d["total_tokens"] > 0, f"{label}: total_tokens should be > 0"


def _test_providers():
    sys_prompt = "You are concise."
    usr_prompt = "Say hello in one short sentence."

    reset_usage()

    _test_provider_usage("OpenAI",      get_provider("gpt-4o-mini"),                          sys_prompt, usr_prompt)
    _test_provider_usage("Gemini",      get_provider("gemini-2.5-flash"),                     sys_prompt, usr_prompt)
    _test_provider_usage("Anthropic",   get_provider("claude-haiku-4-5"),                     sys_prompt, usr_prompt)
    _test_provider_usage("HuggingFace", get_provider("meta-llama/Meta-Llama-3-8B-Instruct"),  sys_prompt, usr_prompt)

    print(f"\n--- Aggregated (4 calls) ---\n{get_usage()}")


if __name__ == "__main__":
    _test_providers()
