"""
Networking helper + LLM label parser.

* ArgoGatewayClient  – simple JSON POST wrapper with retries
* llm_label          – build prompt (template may be injected) & parse
"""

from __future__ import annotations

import logging
import os
import random
import re
import time
from typing import Optional, Tuple

import httpx

from .prompts import LABEL_SET, build_annotation_prompt, get_prompt_template

SYSTEM_MSG = (
    "You are an expert curator of gene functional annotations, similar to specialists at SwissProt or RAST. "
    "Your expertise is in standardizing nomenclature and detecting relationships between biological terms. "
    "When comparing two terms, carefully analyze their semantic relationship according to established ontology principles. "
    "Consider EC numbers, molecular functions, cellular roles, and established conventions in biological databases. "
    "Respond with '<Label> — <detailed explanation with specific evidence from nomenclature patterns, database conventions, and biological function>'. "
    "Your classification should reflect the standards used in curated databases for terminology reconciliation."
)

_ALIAS = {
    "Identical": "Exact",
    "Same": "Exact",
    "Equivalent": "Synonym",
    "Similar": "Synonym",
    "Partial": "Related",
}
_DASH = re.compile(r"\s*[—–-]\s*")

__all__ = ["ArgoGatewayClient", "llm_label"]


# ─────────────────────────────────────────────────────────────────────
class ArgoGatewayClient:
    def __init__(
        self,
        model: str = "gpto3mini",
        env: Optional[str] = None,
        stream: bool = False,
        user: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        retries: int = 5,
        **kwargs,
    ):
        env = env or ("dev" if model.startswith("gpto") else "prod")
        base = (
            f"https://apps{'-' + env if env != 'prod' else ''}"
            ".inside.anl.gov/argoapi/api/v1/resource/"
        )
        self.url = base + ("streamchat/" if stream else "chat/")
        self.model = model
        self.user = user or os.getenv("ARGO_USER") or os.getlogin()
        self.retries = retries
        # Accept extra kwargs (e.g., 'temperature') for vote mode
        self.temperature = kwargs.get("temperature")

        self.headers = {"Content-Type": "application/json"}
        if api_key or os.getenv("ARGO_API_KEY"):
            self.headers["x-api-key"] = api_key or os.getenv("ARGO_API_KEY")

        self.cli = httpx.Client(timeout=timeout, follow_redirects=True)

    # ------------------------------------------------------------------
    def _payload(self, prompt: str, system: str) -> dict:
        if self.model.startswith("gpto"):  # o-series
            return {
                "user": self.user,
                "model": self.model,
                "system": system,
                "prompt": [prompt],
                "max_completion_tokens": 512,
            }
        return {  # GPT-style
            "user": self.user,
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
        }

    # ------------------------------------------------------------------
    def chat(self, prompt: str, *, system: str = "") -> str:
        payload = self._payload(prompt, system)
        for att in range(self.retries + 1):
            r = self.cli.post(self.url, json=payload, headers=self.headers)
            # treat 5xx as retry-able (transient upstream errors)
            if r.status_code >= 500:
                reason = "StatusError"
            else:
                r.raise_for_status()
                reason = ""

            if not reason:
                j = r.json()
                txt = (
                    j["choices"][0]["message"]["content"].strip()
                    if "choices" in j
                    else next(
                        (
                            j[k].strip()
                            for k in ("response", "content", "text")
                            if isinstance(j.get(k), str)
                        ),
                        "",
                    )
                )
                if txt:
                    return txt
                reason = "blank reply"

            if att < self.retries:
                delay = 1.5 * 2**att + random.random()
                print(f"[retry {att+1}/{self.retries}] {reason}; sleeping {delay:.1f}s")
                time.sleep(delay)
        return ""  # gave up

    # ------------------------------------------------------------------
    def ping(self) -> bool:
        try:
            msg = self.chat("Say: Ready to work!")
            return "ready" in msg.lower()
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────────────
def _parse(raw: str) -> tuple[str, str]:
    if not raw:
        return "Unknown", ""
    raw = raw.strip().rstrip(".")

    # Log the raw response for debugging
    logging.debug(f"Raw LLM response: {raw}")

    # More robust splitting that handles different formats
    parts = _DASH.split(raw, maxsplit=1)

    # Get the label, removing any markdown formatting
    label_part = parts[0].strip()
    label_part = label_part.replace("*", "").strip()

    # Extract the first word as the label
    label = label_part.split()[0].capitalize()

    # Get the evidence part after the dash
    note = parts[1].strip() if len(parts) > 1 else ""

    # Apply any label aliases
    label = _ALIAS.get(label, label)

    if label not in LABEL_SET:
        logging.warning(f"Unknown label: {label} from response: {raw}")
        return "Unknown", raw

    return label, note


def llm_label(  # noqa: C901  (assist wrapper)
    ann_a: str,
    ann_b: str,
    client: ArgoGatewayClient,
    *,
    cfg: dict | None = None,  # pass full cfg to honour prompt_ver
    template: str | None = None,
    with_note: bool = False,
) -> Tuple[str, str] | str:
    """
    Build prompt (optionally overriding the template) → call LLM → parse.

    If *template* is None we fall back to cfg["prompt_ver"] or default.
    """
    if template is None:
        template = get_prompt_template(cfg or {})
    prompt = build_annotation_prompt(ann_a, ann_b, template)

    # Always use the system message regardless of model type
    sysmsg = SYSTEM_MSG

    # Log the prompt and system message
    logging.debug(f"Prompt: {prompt}")
    logging.debug(f"System: {sysmsg}")

    # Get the raw response and parse it
    raw_response = client.chat(prompt, system=sysmsg)
    logging.debug(f"Raw LLM response: {raw_response}")

    # Parse the response
    label, note = _parse(raw_response)

    # For pipeline.py, always include the full label (not abbreviated)
    return (label, note) if with_note else label
