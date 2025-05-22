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
from dotenv import load_dotenv

from .prompts import LABEL_SET, build_annotation_prompt, get_prompt_template

# Load environment variables from .env file
load_dotenv()

# SYSTEM_MSG = (
#    "You are an expert curator of gene functional annotations, similar to specialists at SwissProt or RAST. "
#    "Your expertise is in standardizing nomenclature and detecting relationships between biological terms. "
#    "When comparing two terms, carefully analyze their semantic relationship according to established ontology principles. "
#    "Consider EC numbers, molecular functions, cellular roles, and established conventions in biological databases. "
#    "Respond with '<Label> — <detailed explanation with specific evidence from nomenclature patterns, database conventions, and biological function>'. "
#    "Your classification should reflect the standards used in curated databases for terminology reconciliation."
# )
SYSTEM_MSG = (
    "You are an expert curator of gene functional annotations (SwissProt or RAST style)."
    "When you compare two annotations, classify their relationship according to ontology practice "
    "(EC, UniProt, KEGG, phage manuals). "
    "Respond in the format: Label — short justification (evidence or rule number)."
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

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

# Models that exist on both prod & dev – we prefer prod but can fall back
DUAL_ENV_MODELS = {"gpt4o", "gpt4olatest"}

# o-series models (reasoning family) need more time to respond
O_SERIES_TIMEOUT = 120.0  # sec – max per request
DEFAULT_TIMEOUT = 30.0

# HTTP codes indicating the request is still being processed
_PROCESSING = {102, 202}

# Polling settings (used only when we get 102/202)
POLL_EVERY = 3.0  # sec

# ---------------------------------------------------------------------------
# Logging setup (light-weight, only if caller/root logger not configured)
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(_h)
# default INFO – can be flipped to DEBUG via env var or kwargs
logger.setLevel(logging.INFO)


def _extract_job_url(r: httpx.Response, base: str) -> str | None:  # helper
    """Return absolute URL that can be polled for job status/result."""
    if "Location" in r.headers:  # standard HTTP header
        loc = r.headers["Location"]
        return loc if loc.startswith("http") else base + loc.lstrip("/")
    try:
        j = r.json()
    except Exception:
        return None
    if "job_url" in j:
        return j["job_url"]
    if job_id := j.get("job_id"):
        return base + f"status/{job_id}"
    return None


# ─────────────────────────────────────────────────────────────────────
class ArgoGatewayClient:
    def __init__(
        self,
        model: str = "gpto3mini",
        env: Optional[str] = None,
        stream: Optional[bool] = None,
        user: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        retries: int = 5,
        **kwargs,
    ):
        # ------------------------------------------------------------------
        # 1. Decide environment (prod vs dev)
        #    • caller can override via *env*
        #    • o-series default to dev
        #    • dual-env models: try prod first, fall back to dev on first 5xx
        # ------------------------------------------------------------------
        self.model = model
        self.env = env or ("dev" if model.startswith("gpto") else "prod")

        # build helper to re-construct URLs when env flips later
        def _base_url(e: str) -> str:
            return (
                f"https://apps{'-' + e if e != 'prod' else ''}"
                ".inside.anl.gov/argoapi/api/v1/resource/"
            )

        self._base_url_fn = _base_url  # keep for later reuse

        base = _base_url(self.env)

        # Decide streaming endpoint
        _STREAM_CAPABLE = {"gpto3mini"}  # expand when other o-series gain SSE
        self._stream = stream if stream is not None else (model in _STREAM_CAPABLE)
        self.url = base + ("streamchat/" if self._stream else "chat/")

        # ------------------------------------------------------------------
        # 2. Auth & identity
        # ------------------------------------------------------------------
        self.user = user or os.getenv("ARGO_USER") or os.getlogin()
        self.retries = retries

        # optional kwargs (e.g. temperature for vote mode)
        self.temperature = kwargs.get("temperature")

        # Store any extra keyword options (e.g., custom max_completion_tokens)
        self._extra = kwargs

        # ------------------------------------------------------------------
        # 3. Compute HTTP timeout (must be BEFORE first use of self.timeout)
        # ------------------------------------------------------------------
        # Compute timeout: caller override wins; else pick model-based default
        self.timeout = (
            timeout
            if (timeout is not None and timeout > 0)
            else (O_SERIES_TIMEOUT if model.startswith("gpto") else DEFAULT_TIMEOUT)
        )

        # enable verbose logging when ARGO_DEBUG=1 or debug kwarg
        if os.getenv("ARGO_DEBUG") or kwargs.get("debug"):
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug mode active for ArgoGatewayClient")

        logger.info(
            "ArgoGatewayClient initialised | model=%s env=%s timeout=%.1fs url=%s",
            self.model,
            self.env,
            self.timeout,
            self.url,
        )

        # Instantiate HTTP client AFTER we know timeout
        self.cli = httpx.Client(timeout=self.timeout, follow_redirects=True)

        # headers (api key optional)
        self.headers = {"Content-Type": "application/json"}
        if api_key or os.getenv("ARGO_API_KEY"):
            self.headers["x-api-key"] = api_key or os.getenv("ARGO_API_KEY")

        # ------------------------------------------------------------------
        # 4. Dual-env prod→dev quick check (single ping)
        # ------------------------------------------------------------------
        if env is None and model in DUAL_ENV_MODELS and self.env == "prod":
            try:
                if not self.ping():
                    # switch to dev & rebuild URL
                    self.env = "dev"
                    base = _base_url("dev")
                    self.url = base + ("streamchat/" if self._stream else "chat/")
            except Exception:
                # network error → assume prod dead, flip to dev
                self.env = "dev"
                base = _base_url("dev")
                self.url = base + ("streamchat/" if self._stream else "chat/")

    # ------------------------------------------------------------------
    def _payload(self, prompt: str, system: str) -> dict:
        if self.model.startswith("gpto") or self.model.startswith(
            "o"
        ):  # o-series models
            # Allow caller override; else default to a small number (32) to avoid
            # massive completions that sometimes trigger gateway bugs.
            payload = {
                "user": self.user,
                "model": self.model,
                "prompt": [prompt],
            }
            # NEW: only send if explicitly configured
            if "max_completion_tokens" in self._extra:
                payload["max_completion_tokens"] = int(
                    self._extra["max_completion_tokens"]
                )

            # o3 supports system; o1/o1-mini ignore it – harmless to include.
            if system:
                payload["system"] = system
            return payload
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

        # Allow one automatic flip between /chat/ and /streamchat/ on blank reply
        endpoint_switched = False
        sentinel_injected = False

        for att in range(self.retries + 1):
            logger.debug("Attempt %d POST → %s", att + 1, self.url)
            try:
                r = self.cli.post(self.url, json=payload, headers=self.headers)
            except httpx.TimeoutException:
                reason = "timeout"
                logger.warning("Timeout on attempt %d", att + 1)
                r = None
            else:
                reason = ""

            if r is not None:
                logger.debug(
                    "Status %s | body preview: %s", r.status_code, r.text[:120]
                )
                # handling for server responses
                if r.status_code in _PROCESSING:
                    logger.debug(
                        "Processing accepted (HTTP %s); enter poll loop", r.status_code
                    )
                    txt = self._poll_for_result(r)
                    if txt:
                        logger.debug(
                            "Poll loop succeeded; returning text (%d chars)", len(txt)
                        )
                        return txt
                    reason = "processing timeout"
                elif r.status_code >= 500:
                    logger.warning("5xx (%s) on attempt %d", r.status_code, att + 1)
                    # on prod failure & dual-env model, auto-retry once on dev
                    if (
                        self.model in DUAL_ENV_MODELS
                        and self.env == "prod"
                        and att == 0
                    ):
                        self.env = "dev"
                        self.url = self._base_url_fn("dev") + (
                            "streamchat/" if self._stream else "chat/"
                        )
                        continue  # retry immediately on dev
                    reason = f"{r.status_code}"
                else:
                    r.raise_for_status()
                    txt = self._extract_txt(r)
                    if txt:
                        logger.debug("Received final text (%d chars)", len(txt))
                        return txt
                    reason = "blank reply"
                    logger.warning("Blank reply body: %s", r.text[:200])

                    # (1) Flip endpoint once
                    if not endpoint_switched:
                        self._stream = not self._stream
                        base = self._base_url_fn(self.env)
                        self.url = base + ("streamchat/" if self._stream else "chat/")
                        payload = self._payload(prompt, system)
                        endpoint_switched = True
                        logger.info(
                            "Endpoint switched due to blank reply → %s", self.url
                        )
                        continue  # retry immediately without back-off

                    # (2) Inject sentinel once, force /chat/
                    if not sentinel_injected:
                        prompt = f"Label: {prompt}"
                        self._stream = False
                        base = self._base_url_fn(self.env)
                        self.url = base + "chat/"
                        payload = self._payload(prompt, system)
                        sentinel_injected = True
                        logger.info(
                            'Sentinel injected → retry with /chat/ and "Label:" prefix'
                        )
                        continue  # retry again

            if att < self.retries:
                delay = 1.5 * 2**att + random.random()
                print(f"[retry {att+1}/{self.retries}] {reason}; sleeping {delay:.1f}s")
                logger.debug("Retrying after %.1fs due to %s", delay, reason)
                time.sleep(delay)
        logger.error("All attempts exhausted; final failure")
        raise RuntimeError("exhausted retries")

    # ------------------------------------------------------------------
    def ping(self) -> bool:
        try:
            msg = self.chat("Say: Ready to work!")
            return "ready" in msg.lower()
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_txt(self, r: httpx.Response) -> str:
        """Return cleaned text from any successful 200 response."""
        try:
            j = r.json()
        except Exception:
            # If not JSON but body present, treat whole text as the answer
            raw = r.text.strip()
            return raw

        if "choices" in j:
            return j["choices"][0]["message"]["content"].strip()
        for k in ("response", "content", "text"):
            if isinstance(j.get(k), str):
                return j[k].strip()
        # As a final fallback, return the entire trimmed body
        return r.text.strip()

    def _poll_for_result(self, first: httpx.Response) -> str | None:
        """Poll the gateway when we get 102/202 until we receive the final 200."""
        base = self._base_url_fn(self.env)
        poll_url = _extract_job_url(first, base) or first.url
        logger.debug("Start polling at %s", poll_url)
        waited = 0.0
        while waited < self.timeout:
            time.sleep(POLL_EVERY)
            waited += POLL_EVERY
            try:
                r = self.cli.get(poll_url, headers=self.headers)
            except httpx.TimeoutException:
                logger.debug("Timeout while polling (waited %.1fs)", waited)
                continue
            if r.status_code in _PROCESSING:
                logger.debug("Still processing (%s) after %.1fs", r.status_code, waited)
                continue  # still running
            if r.status_code == 200:
                txt = self._extract_txt(r)
                if txt:
                    logger.debug("Polling succeeded in %.1fs", waited)
                    return txt
                return None  # blank – treat as failure
            # any other status ⇒ break & let outer loop retry/backoff
            logger.warning(
                "Unexpected status %s during polling after %.1fs", r.status_code, waited
            )
            break
        logger.warning("Polling timed out after %.1fs", waited)
        return None


# ─────────────────────────────────────────────────────────────────────
def _parse(raw: str) -> tuple[str, str]:
    if not raw:
        return "Unknown", ""
    # Remove leading/trailing whitespace and trailing period
    raw = raw.strip().rstrip(".")

    # Debug raw response
    logging.debug(f"Raw LLM response: {raw}")

    # Split on dash separators
    parts = _DASH.split(raw, maxsplit=1)

    # Clean and extract raw label (strip quotes/punctuation)
    label_part = parts[0].replace("*", "").strip()
    label_raw = label_part.split()[0].strip("\"'")
    # Normalize and apply alias
    label_norm = label_raw.capitalize()
    label_mapped = _ALIAS.get(label_norm, label_norm)
    # Case-insensitive match against known labels
    valid_map = {lbl.lower(): lbl for lbl in LABEL_SET}
    if label_mapped.lower() in valid_map:
        label = valid_map[label_mapped.lower()]
    else:
        logging.warning(f"Unknown label: {label_mapped} from response: {raw}")
        return "Unknown", raw

    # Extract evidence text if present
    note = parts[1].strip() if len(parts) > 1 else ""

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
