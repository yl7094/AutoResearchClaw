"""OpenClaw Zero Token client — free LLM access via browser-authenticated gateway.

This client integrates with OpenClaw Zero Token gateway (https://github.com/linuxhsj/openclaw-zero-token)
which provides free access to ChatGPT, Claude, Gemini, DeepSeek, Qwen, Doubao, Kimi, GLM, Grok, 
Xiaomi MiMo and more through a unified OpenAI-compatible API.

The gateway runs locally (default port 3002) and uses browser-stored credentials from web logins,
eliminating the need for paid API tokens.

Usage:
    1. Start Chrome in debug mode: ./start-chrome-debug.sh
    2. Onboard providers: ./onboard.sh webauth
    3. Start gateway: ./server.sh
    4. Configure ResearchClaw to use provider: "openclaw-zero-token"
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


@dataclass
class OpenClawConfig:
    """Configuration for OpenClaw Zero Token client."""

    base_url: str = "http://127.0.0.1:3002/v1"
    api_key: str = ""
    primary_model: str = "deepseek-web/deepseek-chat"
    fallback_models: list[str] = field(
        default_factory=lambda: [
            "claude-web/claude-sonnet-4-6",
            "qwen-web/qwen-3.5-plus",
            "doubao-web/doubao-seed-2.0",
            "kimi-web/moonshot-v1-128k",
        ]
    )
    max_tokens: int = 4096
    temperature: float = 0.7
    max_retries: int = 3
    retry_base_delay: float = 2.0
    timeout_sec: int = 300
    user_agent: str = _DEFAULT_USER_AGENT
    # Auto-switch between models based on availability
    auto_switch: bool = True
    # List of available models (can be fetched from /v1/models endpoint)
    available_models: list[str] = field(default_factory=list)


class OpenClawClient:
    """Client for OpenClaw Zero Token gateway.
    
    Provides OpenAI-compatible chat completion interface with automatic
    model switching across multiple free LLM providers.
    """

    def __init__(self, config: OpenClawConfig) -> None:
        self.config = config
        self._model_chain = [config.primary_model] + list(config.fallback_models)
        self._base_url = config.base_url.rstrip("/")
        
    @classmethod
    def from_rc_config(cls, rc_config: Any) -> OpenClawClient:
        """Create OpenClawClient from ResearchClaw config.
        
        Args:
            rc_config: ResearchClaw configuration object
            
        Returns:
            Configured OpenClawClient instance
        """
        from researchclaw.llm import PROVIDER_PRESETS
        
        provider = getattr(rc_config.llm, "provider", "openclaw-zero-token")
        preset = PROVIDER_PRESETS.get(provider, {})
        preset_base_url = preset.get("base_url")
        
        import os
        api_key = str(
            rc_config.llm.api_key 
            or os.environ.get(rc_config.llm.api_key_env, "") 
            or ""
        )
        
        # Use preset base_url if available and config doesn't override
        base_url = rc_config.llm.base_url or preset_base_url or "http://127.0.0.1:3002/v1"
        
        # Ensure base_url has /v1 suffix for OpenAI compatibility
        if not base_url.endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        
        config = OpenClawConfig(
            base_url=base_url,
            api_key=api_key,
            primary_model=rc_config.llm.primary_model or "deepseek-web/deepseek-chat",
            fallback_models=list(rc_config.llm.fallback_models or []),
            max_tokens=getattr(rc_config.llm, "max_tokens", 4096),
            temperature=getattr(rc_config.llm, "temperature", 0.7),
            max_retries=getattr(rc_config.llm, "max_retries", 3),
            timeout_sec=getattr(rc_config.llm, "timeout_sec", 300),
        )
        
        return cls(config)
    
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
        system: str | None = None,
        strip_thinking: bool = False,
    ) -> Any:
        """Send a chat completion request with retry and model fallback.
        
        Args:
            messages: List of {role, content} dicts.
            model: Override model (skips fallback chain).
            max_tokens: Override max token count.
            temperature: Override temperature.
            json_mode: Request JSON response format.
            system: Prepend a system message.
            strip_thinking: If True, strip <think>…</think> reasoning
                tags from the response content.
                
        Returns:
            LLMResponse with content and metadata.
        """
        from researchclaw.llm.client import LLMResponse
        
        if system:
            messages = [{"role": "system", "content": system}] + messages
        
        models = [model] if model else self._model_chain
        max_tok = max_tokens or self.config.max_tokens
        temp = temperature if temperature is not None else self.config.temperature
        
        last_error: Exception | None = None
        
        for m in models:
            try:
                resp = self._call_with_retry(m, messages, max_tok, temp, json_mode)
                if strip_thinking:
                    from researchclaw.utils.thinking_tags import strip_thinking_tags
                    
                    resp = LLMResponse(
                        content=strip_thinking_tags(resp.content),
                        model=resp.model,
                        prompt_tokens=resp.prompt_tokens,
                        completion_tokens=resp.completion_tokens,
                        total_tokens=resp.total_tokens,
                        finish_reason=resp.finish_reason,
                        truncated=resp.truncated,
                        raw=resp.raw,
                    )
                return resp
            except Exception as exc:  # noqa: BLE001
                logger.warning("Model %s failed: %s. Trying next.", m, exc)
                last_error = exc
        
        raise RuntimeError(
            f"All models failed. Last error: {last_error}"
        ) from last_error
    
    def _call_with_retry(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> Any:
        """Call with exponential backoff retry."""
        from researchclaw.llm.client import LLMResponse
        
        for attempt in range(self.config.max_retries):
            try:
                return self._raw_call(model, messages, max_tokens, temperature, json_mode)
            except urllib.error.HTTPError as e:
                status = e.code
                body = ""
                try:
                    body = e.read().decode()[:500]
                except Exception:  # noqa: BLE001
                    pass
                
                # Non-retryable errors
                if status == 403 and "not allowed to use model" in body:
                    raise  # Model not available — let fallback handle
                
                # Retryable: 429 (rate limit), 500, 502, 503, 504
                if status in (400, 429, 500, 502, 503, 504):
                    delay = self.config.retry_base_delay * (2**attempt)
                    import random
                    delay += random.uniform(0, delay * 0.3)
                    logger.info(
                        "Retry %d/%d for %s (HTTP %d). Waiting %.1fs.",
                        attempt + 1,
                        self.config.max_retries,
                        model,
                        status,
                        delay,
                    )
                    import time
                    time.sleep(delay)
                    continue
                
                raise  # Other HTTP errors
            except urllib.error.URLError:
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_base_delay * (2**attempt)
                    import time
                    time.sleep(delay)
                    continue
                raise
        
        raise RuntimeError(
            f"LLM call failed after {self.config.max_retries} retries for model {model}"
        )
    
    def _raw_call(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> Any:
        """Make a single API call to OpenClaw gateway."""
        from researchclaw.llm.client import LLMResponse
        
        # Copy messages to avoid mutating caller's list
        msgs = [dict(m) for m in messages]
        
        # Build request body (OpenAI-compatible format)
        body = {
            "model": model,
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        
        payload = json.dumps(body).encode("utf-8")
        url = f"{self._base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "User-Agent": self.config.user_agent,
        }
        
        req = urllib.request.Request(url, data=payload, headers=headers)
        
        try:
            with urllib.request.urlopen(
                req, timeout=self.config.timeout_sec
            ) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, OSError) as exc:
            logger.error("OpenClaw gateway unreachable at %s: %s", url, exc)
            raise
        
        if not isinstance(data, dict):
            raise ValueError(
                f"Malformed API response: expected JSON object, got {type(data).__name__}: {data}"
            )
        
        # Handle API error responses
        if "error" in data and data["error"] is not None:
            error_info = data["error"]
            if isinstance(error_info, dict):
                error_msg = str(error_info.get("message", str(error_info)))
                error_type = str(error_info.get("type", "api_error"))
            else:
                error_msg = str(error_info)
                error_type = "api_error"
            import io
            
            raise urllib.error.HTTPError(
                "",
                500,
                f"{error_type}: {error_msg}",
                None,
                io.BytesIO(error_msg.encode()),
            )
        
        # Parse OpenAI-compatible response
        if "choices" not in data or not data["choices"]:
            raise ValueError(f"Malformed API response: missing choices. Got: {data}")
        
        choice = data["choices"][0]
        usage = data.get("usage", {})
        
        message = choice.get("message", {})
        content = message.get("content") or ""
        
        return LLMResponse(
            content=content,
            model=data.get("model", model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            finish_reason=choice.get("finish_reason", ""),
            truncated=(choice.get("finish_reason", "") == "length"),
            raw=data,
        )
    
    def list_models(self) -> list[str]:
        """Fetch available models from OpenClaw gateway.
        
        Returns:
            List of model identifiers (e.g., "deepseek-web/deepseek-chat")
        """
        url = f"{self._base_url}/models"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "User-Agent": self.config.user_agent,
        }
        
        req = urllib.request.Request(url, headers=headers)
        
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, OSError) as exc:
            logger.warning("Failed to fetch models from OpenClaw: %s", exc)
            return self._model_chain
        
        models = []
        if isinstance(data, dict) and "data" in data:
            for item in data["data"]:
                if isinstance(item, dict) and "id" in item:
                    models.append(item["id"])
        
        if models:
            self.config.available_models = models
            logger.info("Found %d models from OpenClaw gateway", len(models))
        
        return models or self._model_chain
    
    def preflight(self) -> tuple[bool, str]:
        """Quick connectivity check - one minimal chat call.
        
        Returns (success, message).
        """
        try:
            _ = self.chat(
                [{"role": "user", "content": "ping"}],
                max_tokens=64,
                temperature=0,
            )
            return True, f"OK - OpenClaw gateway responding with model {self.config.primary_model}"
        except urllib.error.HTTPError as e:
            status_map = {
                401: "Invalid API key/token",
                403: f"Model {self.config.primary_model} not allowed",
                404: f"Endpoint not found: {self._base_url}",
                429: "Rate limited - try again in a moment",
            }
            msg = status_map.get(e.code, f"HTTP {e.code}")
            return False, msg
        except (urllib.error.URLError, OSError) as e:
            return False, f"Connection failed: {e}. Is OpenClaw gateway running?"
        except RuntimeError as e:
            return False, f"All models failed: {e}"
