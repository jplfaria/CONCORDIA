"""
concord.llm.argo_gateway
------------------------
Networking layer + prompt helper.

• ArgoGatewayClient  –  posts JSON to /resource/chat/ or /streamchat/
• llm_label          –  builds a prompt via prompts.build_annotation_prompt
                         and parses "<Label> — <reason>"
"""

from __future__ import annotations
import os, httpx, re
from typing import Optional, Tuple

from .prompts import build_annotation_prompt, LABEL_SET

__all__ = ["ArgoGatewayClient", "llm_label"]

# ======================================================================
class ArgoGatewayClient:
    """
    Thin wrapper around Argo Gateway API.

    Auto-routing:
        gpto*  → apps-dev
        others → apps-prod
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
        env = env or ("dev" if model.startswith("gpto") else "prod")
        base = (
            f"https://apps{'-' + env if env != 'prod' else ''}."
            "inside.anl.gov/argoapi/api/v1/resource/"
        )
        self.url = base + ("streamchat/" if stream else "chat/")

        self.model = model
        self.user = user or os.getenv("ARGO_USER") or os.getlogin()

        self.headers = {"Content-Type": "application/json"}
        if api_key or os.getenv("ARGO_API_KEY"):
            self.headers["x-api-key"] = api_key or os.getenv("ARGO_API_KEY")

        self.cli = httpx.Client(timeout=timeout, follow_redirects=True)

    # ------------------------------------------------------------------
    def chat(self, prompt: str, *, system: str = "") -> str:
        """Return raw assistant text (stripped)."""
        payload = (
            {  # o-series
                "user": self.user,
                "model": self.model,
                "system": system,
                "prompt": [prompt],
                "max_completion_tokens": 512,
            }
            if self.model.startswith("gpto")
            else {  # GPT-style
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

        # extract assistant text
        if "choices" in data:  # OpenAI style
            return data["choices"][0]["message"]["content"].strip()
        for k in ("response", "content", "text"):
            if k in data and isinstance(data[k], str):
                return data[k].strip()
        return ""

    # ------------------------------------------------------------------
    def ping(self) -> bool:
        try:
            msg = self.chat("Say: Ready to work!")
            print(msg or "<empty>")
            return "ready" in msg.lower()
        except Exception as exc:
            print("Ping failed:", exc)
            return False


# ======================================================================
_DASH = re.compile(r"\s*[—–-]\s*")  # em-dash, en-dash, or hyphen with spaces


def llm_label(
    ann_a: str,
    ann_b: str,
    client: ArgoGatewayClient,
    *,
    with_note: bool = False
) -> Tuple[str, str] | str:
    """
    Build prompt, send to LLM, parse "<Label> — <reason>".

    Returns:
        • (label, note)  when with_note=True
        • label          when with_note=False
    """
    prompt = build_annotation_prompt(ann_a, ann_b)
    reply  = client.chat(prompt).strip()

    if not reply:
        return ("Unknown", "") if with_note else "Unknown"

    # Split on first dash; ensure we always have at least label + note str
    parts = _DASH.split(reply, maxsplit=1)
    label = parts[0].strip().capitalize()
    note  = parts[1].strip() if len(parts) > 1 else ""

    if label not in LABEL_SET:   # fall-back if LLM drifted
        label, note = "Unknown", reply

    return (label, note) if with_note else label