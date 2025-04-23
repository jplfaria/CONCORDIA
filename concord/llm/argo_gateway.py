"""
concord.llm.argo_gateway
------------------------
Resilient Argo Gateway client (April-2025 spec).

Highlights
~~~~~~~~~~
• Auto-routes o-series models (gpto*) to **apps-dev**; GPT-3.5/4/4o to **prod**.  
• Correct payloads
    – o-series : {user, model, system, prompt:[...], max_completion_tokens}  
    – GPT-style: OpenAI messages schema
• Robust extractor works with either OpenAI JSON (choices) **or**
  simple {"response": "..."} gateway replies.
• Graceful handling of empty responses → returns "Unknown".
• `ping()` helper prints **Ready to work!** when connected.
"""

from __future__ import annotations
import os
from typing import Optional

import httpx

__all__ = ["ArgoGatewayClient", "llm_label"]


# ======================================================================
class ArgoGatewayClient:
    """Thin wrapper around Argo Gateway chat & streamchat endpoints."""

    def __init__(
        self,
        model: str = "gpto3mini",
        env: Optional[str] = None,         # auto-select if None
        stream: bool = False,
        user: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        # ---------------- choose environment ---------------------------
        if env is None:
            env = "dev" if model.startswith("gpto") else "prod"

        base = (
            f"https://apps{'-'+env if env!='prod' else ''}.inside.anl.gov/"
            "argoapi/api/v1/resource/"
        )
        endpoint = "streamchat/" if stream else "chat/"
        self.url = f"{base}{endpoint}"          # …/resource/chat/

        # ---------------- common fields -------------------------------
        self.model = model
        self.stream = stream
        self.user = user or os.getenv("ARGO_USER") or os.getlogin()

        self.headers = {"Content-Type": "application/json"}
        if api_key or os.getenv("ARGO_API_KEY"):
            self.headers["x-api-key"] = api_key or os.getenv("ARGO_API_KEY")

        self.cli = httpx.Client(timeout=timeout, follow_redirects=True)

    # ------------------------------------------------------------------
    def _build_payload(self, prompt: str, system: str) -> dict:
        """Return request dict for given model family."""
        if self.model.startswith("gpto"):          # o-series
            return {
                "user": self.user,
                "model": self.model,
                "system": system,
                "prompt": [prompt],
                "max_completion_tokens": 1024
            }
        # GPT-3.5 / 4 / 4o
        return {
            "user": self.user,
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            "temperature": 0.0
        }

    def _extract_content(self, data: dict) -> str:
        """Return assistant text from gateway JSON (handles two formats)."""
        if "choices" in data:  # OpenAI style
            return data["choices"][0]["message"]["content"].strip()
        for key in ("response", "content", "text"):
            val = data.get(key)
            if isinstance(val, str):
                return val.strip()
        return ""  # empty / unrecognised format

    # ------------------------------------------------------------------
    def chat(self, prompt: str, system: str = "You are a precise bio-curator.") -> str:
        payload = self._build_payload(prompt, system)
        r = self.cli.post(self.url, json=payload, headers=self.headers)
        r.raise_for_status()
        return self._extract_content(r.json())

    # ------------------------------------------------------------------
    def ping(self) -> bool:
        """Health-check; prints gateway reply; returns True/False."""
        try:
            reply = self.chat("Say: Ready to work!", system="")
            print(reply or "<empty reply>")
            return "ready" in reply.lower()
        except Exception as exc:
            print("Ping failed:", exc)
            return False


# ======================================================================
def llm_label(a: str, b: str, client: ArgoGatewayClient) -> str:
    """
    Ask the LLM to classify a pair of annotations.
    Returns: Identical | Synonym | Partial | New | Unknown
    """
    prompt = (
        "Classify the relationship between these two gene/protein function "
        "annotations. Respond with ONE word only from "
        "[Identical, Synonym, Partial, New].\n"
        f"A: {a}\nB: {b}"
    )
    reply = client.chat(prompt).strip()

    # Graceful fallback for blank / unexpected replies
    if not reply:
        return "Unknown"

    return reply.split()[0].capitalize()