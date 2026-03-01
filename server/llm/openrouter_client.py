"""OpenRouter client wrapper with retries, fallbacks, and JSON parsing helpers."""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

from openai import OpenAI


@dataclass
class LLMResponse:
    content: str
    model: str
    degraded: bool


class OpenRouterClient:
    """Small OpenRouter abstraction for robust text/json chat calls."""

    def __init__(
        self,
        api_key: str,
        primary_model: str,
        fallback_models: Optional[list[str]] = None,
        timeout: float = 20.0,
        app_name: str = "Real-Eyes",
        referer: Optional[str] = None,
        max_retries: int = 2,
    ):
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=timeout,
        )
        self.primary_model = primary_model
        self.fallback_models = [m for m in (fallback_models or []) if m and m != primary_model]
        self.max_retries = max(0, int(max_retries))
        self.timeout = timeout
        self.app_name = app_name
        self.referer = referer or os.environ.get("OPENROUTER_HTTP_REFERER", "https://real-eyes.local")

        self.last_model = primary_model
        self.degraded_mode = False

    @staticmethod
    def _normalize_content(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for entry in content:
                if isinstance(entry, dict):
                    text = entry.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts).strip()
        return str(content).strip()

    @staticmethod
    def _extract_json_blob(text: str) -> Optional[str]:
        if not text:
            return None

        match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
        if match:
            return match.group(1)

        start = text.find("{")
        if start < 0:
            return None

        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]

        return None

    def _build_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        history: Optional[list[dict[str, Any]]] = None,
        images_b64: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history:
            for item in history:
                role = item.get("role")
                content = item.get("content")
                if role in {"user", "assistant", "system"} and isinstance(content, str):
                    messages.append({"role": role, "content": content})

        if images_b64:
            content_parts: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
            for img_b64 in images_b64:
                if not img_b64:
                    continue
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}",
                        },
                    }
                )
            messages.append({"role": "user", "content": content_parts})
        else:
            messages.append({"role": "user", "content": user_prompt})

        return messages

    def _request_once(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_headers={
                "HTTP-Referer": self.referer,
            },
        )
        msg = response.choices[0].message
        return self._normalize_content(msg.content)

    def _retry_with_backoff(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._request_once(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:  # pragma: no cover - provider/network variance
                err = exc
                if attempt < self.max_retries:
                    sleep_s = (0.25 * (2 ** attempt)) + random.uniform(0.0, 0.2)
                    time.sleep(sleep_s)
        if err is None:
            raise RuntimeError("Unknown OpenRouter error")
        raise err

    def chat_text(
        self,
        system_prompt: str,
        user_prompt: str,
        history: Optional[list[dict[str, Any]]] = None,
        images_b64: Optional[list[str]] = None,
        temperature: float = 0.4,
        max_tokens: int = 512,
    ) -> LLMResponse:
        models = [self.primary_model] + self.fallback_models
        messages = self._build_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            history=history,
            images_b64=images_b64,
        )

        errors: list[str] = []
        for idx, model in enumerate(models):
            try:
                content = self._retry_with_backoff(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                degraded = idx > 0
                self.last_model = model
                self.degraded_mode = degraded
                return LLMResponse(content=content, model=model, degraded=degraded)
            except Exception as exc:  # pragma: no cover - provider/network variance
                errors.append(f"{model}: {exc}")

        raise RuntimeError("OpenRouter chat failed across all models: " + " | ".join(errors))

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        history: Optional[list[dict[str, Any]]] = None,
        images_b64: Optional[list[str]] = None,
        temperature: float = 0.3,
        max_tokens: int = 768,
    ) -> tuple[dict[str, Any], LLMResponse]:
        response = self.chat_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            history=history,
            images_b64=images_b64,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        blob = self._extract_json_blob(response.content)
        if blob is None:
            raise RuntimeError("No JSON object found in LLM response")
        parsed = json.loads(blob)
        if not isinstance(parsed, dict):
            raise RuntimeError("Expected top-level JSON object from LLM response")
        return parsed, response

    def chat_json_model(
        self,
        model_cls,
        system_prompt: str,
        user_prompt: str,
        history: Optional[list[dict[str, Any]]] = None,
        images_b64: Optional[list[str]] = None,
        temperature: float = 0.3,
        max_tokens: int = 768,
    ):
        parsed, response = self.chat_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            history=history,
            images_b64=images_b64,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return model_cls.model_validate(parsed), response
