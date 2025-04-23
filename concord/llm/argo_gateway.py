"""
concord.llm.argo_gateway  –  resilient Argo client
"""

from __future__ import annotations
import os, httpx
from typing import Optional, Callable, List

__all__ = ["ArgoGatewayClient", "llm_label"]


class ArgoGatewayClient:
    _o_schema: Optional[int] = None                # cache winning schema

    def __init__(
        self,
        model: str = "gpto3mini",
        env: str = "prod",
        stream: bool = False,
        user: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        base = (
            f"https://apps{'' if env=='prod' else f'-{env}'}.inside.anl.gov/"
            "argoapi/api/v1/resource"
        )
        endpoint = "streamchat/" if stream else "chat/"
        self.url = f"{base}/{endpoint}"            # …/resource/chat/

        self.model, self.stream = model, stream
        self.user = user or os.getenv("ARGO_USER") or os.getlogin()

        self.headers = {"Content-Type": "application/json"}
        if api_key or os.getenv("ARGO_API_KEY"):
            self.headers["x-api-key"] = api_key or os.getenv("ARGO_API_KEY")

        self.cli = httpx.Client(timeout=timeout, follow_redirects=True)

    # ----------------------------------------------------------- public
    def chat(self, prompt: str, system: str = "You are a precise bio-curator.") -> str:
        """Return assistant content for a single prompt."""
        if not self.model.startswith("gpto"):
            return self._post(self._payload_gpt(prompt, system))

        builders: List[Callable[[str, str], dict]] = [
            self._o_v1,  # messages + max_tokens
            self._o_v2,  # prompt list + max_tokens
            self._o_v3,  # prompt list + max_completion_tokens
            self._o_v4,  # prompt string + system_prompt + max_tokens
        ]

        if self._o_schema is not None:
            return self._post(builders[self._o_schema](prompt, system))

        for idx, build in enumerate(builders):
            try:
                reply = self._post(build(prompt, system))
                self._o_schema = idx
                return reply
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (400, 422):
                    continue
                raise
        raise RuntimeError("Argo gateway rejected all o-series payload variants.")

    def ping(self) -> bool:
        """Prints gateway reply; returns True on success."""
        try:
            reply = self.chat("Say: Ready to work!", system="")
            print(reply)
            return "ready" in reply.lower()
        except Exception as exc:
            print("Ping failed:", exc)
            return False

    # ---------------------------------------------------- internal helpers
    def _post(self, payload: dict) -> str:
        r = self.cli.post(self.url, json=payload, headers=self.headers)
        r.raise_for_status()
        if self.stream:
            return "".join(
                ch["choices"][0]["delta"]["content"]
                for ch in r.iter_lines()
                if ch
            )
        return r.json()["choices"][0]["message"]["content"].strip()

    # GPT-4/3.5 style
    def _payload_gpt(self, prompt: str, system: str) -> dict:
        return {
            "user": self.user,
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
        }

    # ---------- o-series candidate schemas ----------
    def _o_v1(self, prompt: str, system: str) -> dict:
        return {
            "user": self.user,
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 1024,
        }

    def _o_v2(self, prompt: str, _sys: str) -> dict:
        return {
            "user": self.user,
            "model": self.model,
            "prompt": [prompt],
            "max_tokens": 1024,
        }

    def _o_v3(self, prompt: str, _sys: str) -> dict:
        return {
            "user": self.user,
            "model": self.model,
            "prompt": [prompt],
            "max_completion_tokens": 1024,
        }

    def _o_v4(self, prompt: str, system: str) -> dict:
        return {
            "user": self.user,
            "model": self.model,
            "prompt": prompt,           # single string
            "system_prompt": system,
            "max_tokens": 1024,
        }


# -------------------------------------------------------- helper
def llm_label(a: str, b: str, client: ArgoGatewayClient) -> str:
    prompt = (
        "Classify the relationship between these two gene/protein function "
        "annotations. Respond with one word: Identical, Synonym, Partial, or New.\n"
        f"A: {a}\nB: {b}"
    )
    return client.chat(prompt).split()[0]