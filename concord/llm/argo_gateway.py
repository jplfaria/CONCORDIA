from __future__ import annotations
import os, time, random, httpx, re
from typing import Optional, Tuple

from .prompts import build_annotation_prompt, LABEL_SET, PROMPT_VER

SYSTEM_MSG = (
    "You are a bioinformatics assistant. "
    "Only reply with '<Label> — <very short reason>'."
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


class ArgoGatewayClient:
    def __init__(
        self,
        model: str = "gpto3mini",
        env: Optional[str] = None,
        stream: bool = False,
        user: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        retries: int = 5,                 # 🆕  default heavy-duty retry
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

        self.headers = {"Content-Type": "application/json"}
        if api_key or os.getenv("ARGO_API_KEY"):
            self.headers["x-api-key"] = api_key or os.getenv("ARGO_API_KEY")

        self.cli = httpx.Client(timeout=timeout, follow_redirects=True)

    # ..................................................................
    def _payload(self, prompt: str, system: str) -> dict:
        if self.model.startswith("gpto"):
            return {
                "user": self.user,
                "model": self.model,
                "system": system,
                "prompt": [prompt],
                "max_completion_tokens": 512,
            }
        return {
            "user": self.user,
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
        }

    # ..................................................................
    def chat(self, prompt: str, *, system: str = "") -> str:
        payload = self._payload(prompt, system)
        for att in range(self.retries + 1):
            r = self.cli.post(self.url, json=payload, headers=self.headers)
            r.raise_for_status()
            j = r.json()

            txt = (
                j["choices"][0]["message"]["content"].strip()
                if "choices" in j
                else next((j[k].strip() for k in ("response", "content", "text") if k in j and isinstance(j[k], str)), "")
            )
            if txt:
                return txt

            if att < self.retries:                       # back-off
                delay = 1.5 * 2**att + random.random()
                print(f"[retry {att+1}/{self.retries}] blank reply; sleep {delay:.1f}s")
                time.sleep(delay)
        return ""

    # ..................................................................
    def ping(self) -> bool:
        try:
            return "ready" in self.chat("Say: Ready to work!").lower()
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────────────
def _parse(raw: str) -> tuple[str, str]:
    if not raw:
        return "Unknown", ""
    raw = raw.strip().rstrip(".")
    parts = _DASH.split(raw, maxsplit=1)
    label = parts[0].split()[0].capitalize()
    note = parts[1].strip() if len(parts) > 1 else ""
    label = _ALIAS.get(label, label)
    if label not in LABEL_SET:
        return "Unknown", raw
    return label, note


def llm_label(
    ann_a: str,
    ann_b: str,
    client: ArgoGatewayClient,
    *,
    with_note: bool = False,
) -> Tuple[str, str] | str:
    prompt = build_annotation_prompt(ann_a, ann_b)
    sysmsg = "" if client.model.startswith("gpto") else SYSTEM_MSG
    label, note = _parse(client.chat(prompt, system=sysmsg))
    return (label, note) if with_note else label