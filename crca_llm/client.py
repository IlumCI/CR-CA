"""OpenAI-compatible client for crca_llm (chat completions only)."""

from __future__ import annotations

import os
from dataclasses import dataclass
import time
import random
from loguru import logger
from typing import Any, Dict, List, Optional

import requests


class MissingApiKeyError(RuntimeError):
    pass


@dataclass
class OpenAIClient:
    base_url: str = "https://api.openai.com"
    api_key: Optional[str] = None
    default_model: str = "gpt-4o-mini"
    max_retries: int = 3
    timeout_seconds: int = 60
    enable_audit_log: bool = True

    @classmethod
    def from_env(cls) -> "OpenAIClient":
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
        api_key = os.environ.get("OPENAI_API_KEY")
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        return cls(base_url=base_url, api_key=api_key, default_model=model)

    def _require_key(self) -> None:
        if not self.api_key:
            raise MissingApiKeyError("OPENAI_API_KEY is required for LLM calls.")

    def chat_completion(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 800,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Send a chat completion request and return the text response."""
        self._require_key()
        url = self.base_url.rstrip("/") + "/v1/chat/completions"
        payload: Dict[str, Any] = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        attempt = 0
        while True:
            try:
                if self.enable_audit_log:
                    logger.info(
                        "LLM request model={} tokens={} messages={}",
                        payload["model"],
                        max_tokens,
                        len(messages),
                    )
                resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout_seconds)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except requests.RequestException as exc:
                attempt += 1
                if attempt > self.max_retries:
                    raise
                # exponential backoff with jitter
                sleep_s = (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                time.sleep(sleep_s)

