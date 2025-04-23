"""
concord.llm.argo_gateway
========================
Robust client for Argonne’s Argo Gateway API.

Features
--------
* Auto-select environment: o-series → apps-dev, GPT-3.5/4/4o → prod.
* Correct payloads:
    – o-series  : {user, model, system, prompt:[...], max_completion_tokens}
    – GPT-style : OpenAI messages schema
* Works whether gateway returns OpenAI-style JSON (choices) *or*
  simple {"response": "..."} wrappers.
* Includes `ping()` health-check that prints **Ready to work!**.
"""

from __future__ import annotations
import os
from typing import Optional

import httpx

__all__ = ["ArgoGatewayClient", "llm_label"]


# ---------------------------------------------------------------------- #
class ArgoGatewayClient:
    def __init__(
        self,
        model: str = "gpto3mini",
        env: Optional[str] = None,   # auto-detect if None
        stream: bool = False,
        user: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        # -------- choose gateway environment ---------------------------
        if env is None:
            env = "dev" if model.startswith("gpto") else "prod"

        base = (
            f"https://apps{'-'+env if env!='prod' else ''}.inside.anl.gov/"
            "argoapi/api/v1/resource/"
        )
        endpoint = "streamchat/" if stream else "chat/"
        self.url = f"{base}{endpoint}"          # …/resource/chat/

        # -------- common fields ----------------------------------------
        self.model = model
        self.stream = stream
        self.user = user or os.getenv("ARGO_USER") or os.getlogin()

        self.headers = {"Content-Type": "application/json"}
        if api_key or os.getenv("ARGO_API_KEY"):
            self.headers["x-api-key"] = api_key or os.getenv("ARGO_API_KEY")

        self.cli = httpx.Client(timeout=timeout, follow_redirects=True)

    # ================================================================== #
    # Public methods
    # ------------------------------------------------------------------ #
    def chat(self, prompt: str, system: str = "You are a precise bio-curator.") -> str:
        """Return assistant text for a single prompt."""
        payload = self._build_payload(prompt, system)
        resp = self.cli.post(self.url, json=payload, headers=self.headers)
        resp.raise_for_status()
        return self._extract_content(resp.json())

    def ping(self) -> bool:
        """Health-check; prints reply and returns True/False."""
        try:
            reply = self.chat("Say: Ready to work!", system="")
            print(reply)
            return "ready" in reply.lower()
        except Exception as exc:
            print("Ping failed:", exc)
            return False

    # ================================================================== #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _build_payload(self, prompt: str, system: str) -> dict:
        """Construct request JSON depending on model family."""
        if self.model.startswith("gpto"):          # o-series
            return {
                "user": self.user,
                "model": self.model,
                "system": system,
                "prompt": [prompt],               # list of strings
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
        """Handle both OpenAI and simple gateway response formats."""
        if "choices" in data:                       # OpenAI-style
            return data["choices"][0]["message"]["content"].strip()
        for key in ("response", "content", "text"):
            if key in data and isinstance(data[key], str):
                return data[key].strip()
        raise KeyError("Assistant text not found in gateway JSON", data)


# ---------------------------------------------------------------------- #
def llm_label(a: str, b: str, client: ArgoGatewayClient) -> str:
    """
    Helper used by Concordia pipeline.
    Returns one of: Identical / Synonym / Partial / New
    """
    prompt = (
        "Classify the relationship between these two gene/protein function "
        "annotations. Respond with ONE word: Identical, Synonym, Partial, or New.\n"
        f"A: {a}\nB: {b}"
    )
    return client.chat(prompt).split()[0]