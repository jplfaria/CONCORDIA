# concord/llm/argo_gateway.py
"""
Networking layer for Concordia → Argo Gateway.

• ArgoGatewayClient.chat()      → one-shot prompt with heavy-duty retry
• llm_label()                   → convenience helper returning <label, note>
"""

from __future__ import annotations

import os
import random
import re
import time
from typing import Optional, Tuple

import httpx

from .prompts import build_annotation_prompt, LABEL_SET

# ──────────────────────────────────────────
_SYSTEM_MSG = (
    "You are a bioinformatics assistant. "
    "Reply **only** with '<Label> — <very short reason>'."
)

_ALIAS = {
    "Identical":  "Exact",
    "Same":       "Exact",
    "Equivalent": "Synonym",
    "Similar":    "Synonym",
    "Partial":    "Related",
}
_DASH = re.compile(r"\s*[—–-]\s*")

__all__ = ["ArgoGatewayClient", "llm_label"]

# ──────────────────────────────────────────
class ArgoGatewayClient:
    """
    Tiny wrapper around the ANL Argo Gateway.

    • `model` — gateway model name, e.g. **gpto3mini**, **gpt4o** …
    • `env`   — "dev" or "prod" (default: auto ⇒ dev for o-series, prod otherwise)
    • `stream`— if True route to `/streamchat/`
    • `retries`— how many times to retry on blank reply, 5xx or time-out
    """

    def __init__(
        self,
        model: str = "gpto3mini",
        *,
        env: str | None = None,
        stream: bool = False,
        user: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
        retries: int = 5,
    ) -> None:
        env = env or ("dev" if model.startswith("gpto") else "prod")
        root = (
            f"https://apps{'-' + env if env != 'prod' else ''}"
            ".inside.anl.gov/argoapi/api/v1/resource/"
        )
        self.url = root + ("streamchat/" if stream else "chat/")
        self.model = model
        self.user = user or os.getenv("ARGO_USER") or os.getlogin()
        self.retries = retries

        self.headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key or os.getenv("ARGO_API_KEY"):
            self.headers["x-api-key"] = api_key or os.getenv("ARGO_API_KEY")

        self.cli = httpx.Client(timeout=timeout, follow_redirects=True)

    # ────────────────────────────────────────
    def _payload(self, prompt: str, system: str) -> dict:
        """Return the JSON body expected by each model family."""
        if self.model.startswith("gpto"):          # o-series
            return {
                "user":   self.user,
                "model":  self.model,
                "system": system,
                "prompt": [prompt],
                "max_completion_tokens": 512,
            }
        # OpenAI-style models
        return {
            "user":  self.user,
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            "temperature": 0.0,
        }

    # ────────────────────────────────────────
    def chat(self, prompt: str, *, system: str = "") -> str:
        """
        Send one prompt and return assistant text (stripped).

        Retries up to `self.retries` when:
        • HTTP 5xx        • network timeout        • blank/empty reply
        """
        payload = self._payload(prompt, system)

        for att in range(self.retries + 1):
            try:
                r = self.cli.post(self.url, json=payload, headers=self.headers)
                # treat 5xx as retryable
                if r.status_code >= 500:
                    raise httpx.HTTPStatusError("5xx", request=r.request, response=r)

                r.raise_for_status()
                data = r.json()
            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.HTTPStatusError):
                reason = "Timeout" if isinstance(_, (httpx.ReadTimeout, httpx.ConnectTimeout)) else "StatusError"
                if att < self.retries:
                    delay = 1.5 * 2 ** att + random.random()
                    print(f"[retry {att+1}/{self.retries}] {reason}; sleeping {delay:.1f}s")
                    time.sleep(delay)
                    continue
                return ""            # exhausted retries

            # ─ extract assistant text ─
            txt = ""
            if "choices" in data:                                    # OpenAI schema
                txt = data["choices"][0]["message"]["content"].strip()
            else:                                                    # o-series
                for k in ("response", "content", "text"):
                    if isinstance(data.get(k), str):
                        txt = data[k].strip()
                        break

            if txt:
                return txt                                           # success

            # blank — retry
            if att < self.retries:
                delay = 1.5 * 2 ** att + random.random()
                print(f"[retry {att+1}/{self.retries}] blank reply; sleeping {delay:.1f}s")
                time.sleep(delay)

        return ""  # still blank after all tries

    # ────────────────────────────────────────
    def ping(self) -> bool:
        """Return True if gateway answers at all (case-insensitive 'ready')."""
        try:
            return "ready" in self.chat("Say: Ready to work!").lower()
        except Exception:
            return False


# ════════════════════════════════════════════════════════════════════
def _parse(raw: str) -> tuple[str, str]:
    """Parse '<Label> — <reason>', including legacy-token rescue."""
    if not raw:
        return "Unknown", ""

    raw = raw.strip().rstrip(".")
    parts = _DASH.split(raw, maxsplit=1)

    lbl  = parts[0].split()[0].capitalize()
    note = parts[1].strip() if len(parts) > 1 else ""

    lbl = _ALIAS.get(lbl, lbl)
    if lbl not in LABEL_SET:
        return "Unknown", raw
    return lbl, note


def llm_label(
    ann_a: str,
    ann_b: str,
    client: ArgoGatewayClient,
    *,
    with_note: bool = False,
) -> Tuple[str, str] | str:
    """Helper: build prompt → ask gateway → parse answer."""
    prompt = build_annotation_prompt(ann_a, ann_b)
    sysmsg = "" if client.model.startswith("gpto") else _SYSTEM_MSG

    label, note = _parse(client.chat(prompt, system=sysmsg))
    return (label, note) if with_note else label