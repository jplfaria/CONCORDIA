"""
concord.llm.argo_gateway
------------------------
Robust client for Argonne’s Argo Gateway + helper to classify annotation pairs.
"""

from __future__ import annotations
import os, httpx, re
from typing import Optional, Tuple

__all__ = ["ArgoGatewayClient", "llm_label"]

# ======================================================================
class ArgoGatewayClient:
    """
    Thin wrapper around Argo Gateway chat endpoints.
    Auto-routes: gpto* models → apps-dev ; GPT-3.5/4* → prod
    """

    def __init__(
        self,
        model: str = "gpto3mini",
        env: Optional[str] = None,
        stream: bool = False,
        user: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        if env is None:
            env = "dev" if model.startswith("gpto") else "prod"

        base = (
            f"https://apps{'-'+env if env!='prod' else ''}.inside.anl.gov/"
            "argoapi/api/v1/resource/"
        )
        self.url = base + ("streamchat/" if stream else "chat/")

        self.model = model
        self.user  = user or os.getenv("ARGO_USER") or os.getlogin()
        self.headers = {"Content-Type": "application/json"}
        if api_key or os.getenv("ARGO_API_KEY"):
            self.headers["x-api-key"] = api_key or os.getenv("ARGO_API_KEY")

        self.cli = httpx.Client(timeout=timeout, follow_redirects=True)

    # ------------- core request -------------------
    def chat(self, prompt: str, system: str = "") -> str:
        payload = (
            {  # o-series format
                "user": self.user,
                "model": self.model,
                "system": system,
                "prompt": [prompt],
                "max_completion_tokens": 512,
            }
            if self.model.startswith("gpto")
            else {  # OpenAI messages format
                "user": self.user,
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                "temperature": 0.0,
            }
        )

        r = self.cli.post(self.url, json=payload, headers=self.headers)
        r.raise_for_status()
        data = r.json()
        # openai-style
        if "choices" in data:
            return data["choices"][0]["message"]["content"].strip()
        # simple field
        for k in ("response", "content", "text"):
            if k in data and isinstance(data[k], str):
                return data[k].strip()
        return ""

    # ------------- health check -------------------
    def ping(self) -> bool:
        try:
            reply = self.chat("Say: Ready to work!")
            print(reply or "<empty>")
            return "ready" in reply.lower()
        except Exception as e:
            print("Ping failed:", e)
            return False


# ======================================================================
DASH_RE = re.compile(r"\s*[–—-]\s*")    # match em/en/normal dash with spaces

def llm_label(
    ann_a: str,
    ann_b: str,
    client: ArgoGatewayClient,
    *,
    with_note: bool = False
) -> Tuple[str, str] | str:
    """
    Ask the LLM for a one-word relationship + a brief reason.

    Returns:
        • (label, note)  when with_note=True
        • label          when with_note=False
    """
    prompt = (
        "Classify the relationship between these two gene/protein function "
        "annotations. Respond **on a single line** exactly as:\n"
        "<Label> — <very short reason>\n"
        "Allowed labels: Identical, Synonym, Partial, New.\n\n"
        f"A: {ann_a}\nB: {ann_b}"
    )

    reply = client.chat(prompt).strip()
    if not reply:
        return ("Unknown", "") if with_note else "Unknown"

    # Split on first dash
    parts = DASH_RE.split(reply, maxsplit=1)
    label = parts[0].strip().capitalize()
    note  = parts[1].strip() if len(parts) > 1 else ""

    if with_note:
        return label, note
    return label