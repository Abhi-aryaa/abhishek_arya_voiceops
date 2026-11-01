"""Unified LLM API client supporting Gemini, OpenAI, and Groq with grounded offline synthesis."""
import json
import os
import urllib.error
import urllib.request
from typing import Dict, List, Optional


class LLMClient:
    """Calls hosted LLM APIs (Gemini, OpenAI, Groq) when keys are configured in environment."""

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self.gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
        self.openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
        self.groq_key = os.environ.get("GROQ_API_KEY", "").strip()
        self.provider = (provider or os.environ.get("LLM_PROVIDER", "auto")).lower()
        self.model = model or os.environ.get("LLM_MODEL", "")

        if self.provider == "auto":
            if self.gemini_key:
                self.provider = "gemini"
            elif self.openai_key:
                self.provider = "openai"
            elif self.groq_key:
                self.provider = "groq"
            else:
                self.provider = "local_grounded"

    @property
    def is_live_api_configured(self) -> bool:
        return self.provider in ("gemini", "openai", "groq") and bool(
            self.gemini_key or self.openai_key or self.groq_key
        )

    def generate(self, system_prompt: str, messages: List[Dict[str, str]], temperature: float = 0.0) -> Optional[str]:
        """Call the configured LLM API. Returns None if offline or no API key is set."""
        if not self.is_live_api_configured:
            return None
        try:
            if self.provider == "gemini" and self.gemini_key:
                return self._call_gemini(system_prompt, messages, temperature)
            if self.provider == "openai" and self.openai_key:
                return self._call_openai_compatible(
                    "https://api.openai.com/v1/chat/completions",
                    self.openai_key,
                    self.model or "gpt-4o-mini",
                    system_prompt,
                    messages,
                    temperature,
                )
            if self.provider == "groq" and self.groq_key:
                return self._call_openai_compatible(
                    "https://api.groq.com/openai/v1/chat/completions",
                    self.groq_key,
                    self.model or "llama-3.3-70b-versatile",
                    system_prompt,
                    messages,
                    temperature,
                )
        except Exception:
            return None
        return None

    def _call_gemini(self, system_prompt: str, messages: List[Dict[str, str]], temperature: float) -> Optional[str]:
        model_name = self.model or "gemini-2.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.gemini_key}"
        contents = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
            "generationConfig": {"temperature": temperature, "maxOutputTokens": 400},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()

    def _call_openai_compatible(
        self,
        url: str,
        api_key: str,
        model_name: str,
        system_prompt: str,
        messages: List[Dict[str, str]],
        temperature: float,
    ) -> Optional[str]:
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        payload = {
            "model": model_name,
            "messages": full_messages,
            "temperature": temperature,
            "max_tokens": 400,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
