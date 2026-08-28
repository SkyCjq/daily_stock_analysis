# -*- coding: utf-8 -*-
"""LiteLLM generation backend wrapper."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple

from src.llm.generation_backend import (
    GenerationBackend,
    GenerationCapabilities,
    GenerationResult,
)


logger = logging.getLogger(__name__)

LiteLLMCallable = Callable[..., Tuple[str, str, Dict[str, Any]]]


def _provider_from_model(model: str) -> str:
    if not model:
        return ""
    if "/" in model:
        return model.split("/", 1)[0]
    return "openai"


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _sanitize_gcp_error(exc: Exception) -> str:
    """Redact configured Google API keys from an SDK exception string."""
    text = str(exc)
    for name in ("GCP_GEMINI_API_KEY", "GOOGLE_API_KEY"):
        secret = (os.getenv(name) or "").strip()
        if secret:
            text = text.replace(secret, "***")
    return text


class LiteLLMGenerationBackend(GenerationBackend):
    """Thin adapter around the existing LiteLLM analyzer call path."""

    backend_id = "litellm"
    capabilities = GenerationCapabilities(
        supports_json=True,
        supports_tools=True,
        supports_stream=True,
        supports_vision=False,
        supports_health_check=False,
        supports_smoke_test=False,
    )

    def __init__(self, completion_callable: LiteLLMCallable):
        self._completion_callable = completion_callable

    def _generate_gcp_agent_platform(
        self,
        prompt: str,
        generation_config: Dict[str, Any],
        *,
        system_prompt: Optional[str] = None,
        response_validator: Optional[Callable[[str], None]] = None,
    ) -> Optional[GenerationResult]:
        """Try Google Cloud Gemini Enterprise Agent Platform before LiteLLM.

        This route is deliberately fail-open. The caller catches any
        configuration, authentication, quota, transport, empty-response, or
        response-validation error and continues through the existing LiteLLM
        model chain unchanged.
        """
        if not _env_enabled("GCP_AGENT_ENABLED"):
            return None

        api_key = (
            (os.getenv("GCP_GEMINI_API_KEY") or "").strip()
            or (os.getenv("GOOGLE_API_KEY") or "").strip()
        )
        if not api_key:
            raise RuntimeError("GCP_GEMINI_API_KEY is not configured")

        # Google currently documents the standard Agent Platform API-key path as
        # GOOGLE_API_KEY + GOOGLE_GENAI_USE_ENTERPRISE=True, without
        # GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION. Supplying project/location
        # would switch the SDK toward ADC auth and override the API-key path.
        os.environ.setdefault("GOOGLE_API_KEY", api_key)
        os.environ.setdefault("GOOGLE_GENAI_USE_ENTERPRISE", "True")

        model = (
            (os.getenv("GCP_AGENT_MODEL") or "gemini-3.5-flash").strip()
            or "gemini-3.5-flash"
        )
        max_tokens = (
            generation_config.get("max_output_tokens")
            or generation_config.get("max_tokens")
            or 8192
        )

        from google import genai
        from google.genai import types

        logger.info(
            "[GCP Agent Platform] Priority 0 调用 %s (auth=google_cloud_api_key)",
            model,
        )
        client = genai.Client(
            http_options=types.HttpOptions(api_version="v1"),
        )
        try:
            config_kwargs: Dict[str, Any] = {
                "max_output_tokens": int(max_tokens),
            }
            if system_prompt:
                config_kwargs["system_instruction"] = system_prompt

            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        finally:
            client.close()

        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise ValueError("GCP Agent Platform returned empty response")

        if response_validator is not None:
            response_validator(text)

        metadata = getattr(response, "usage_metadata", None)
        usage = {
            "prompt_tokens": int(
                getattr(metadata, "prompt_token_count", 0) or 0
            ),
            "completion_tokens": int(
                getattr(metadata, "candidates_token_count", 0) or 0
            ),
            "total_tokens": int(
                getattr(metadata, "total_token_count", 0) or 0
            ),
            "provider": "gcp_agent_platform",
        }
        model_used = f"gcp_agent/{model}"
        logger.info("[GCP Agent Platform] %s 响应成功", model_used)

        return GenerationResult(
            text=text,
            model=model_used,
            provider="gcp_agent_platform",
            backend=self.backend_id,
            usage=usage,
            raw=None,
            diagnostics={"priority_route": "gcp_agent_platform"},
        )

    def generate(
        self,
        prompt: str,
        generation_config: Dict[str, Any],
        *,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        stream_progress_callback: Optional[Callable[[int], None]] = None,
        response_validator: Optional[Callable[[str], None]] = None,
        audit_context: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        try:
            gcp_result = self._generate_gcp_agent_platform(
                prompt,
                generation_config,
                system_prompt=system_prompt,
                response_validator=response_validator,
            )
            if gcp_result is not None:
                return gcp_result
        except Exception as exc:
            logger.warning(
                "[GCP Agent Platform] Priority 0 调用失败，"
                "继续原 LiteLLM fallback 链: %s: %s",
                type(exc).__name__,
                _sanitize_gcp_error(exc),
            )

        text, model, usage = self._completion_callable(
            prompt,
            generation_config,
            system_prompt=system_prompt,
            stream=stream,
            stream_progress_callback=stream_progress_callback,
            response_validator=response_validator,
            audit_context=audit_context,
        )
        provider = str((usage or {}).get("provider") or _provider_from_model(model))
        return GenerationResult(
            text=text,
            model=model,
            provider=provider,
            backend=self.backend_id,
            usage=usage or {},
            raw=None,
            diagnostics={},
        )
