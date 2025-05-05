# concord/modes.py
import logging
import re
from typing import TYPE_CHECKING, Any, Tuple

if TYPE_CHECKING:
    pass

from .embedding import cosine_sim, embed_sentence
from .llm.argo_gateway import ArgoGatewayClient
from .llm.prompt_builder import FALLBACK_TEMPLATE, build_annotation_prompt
from .llm.template_store import get_prompt_template

logger = logging.getLogger(__name__)


def annotate_local(a: str, b: str, cfg: Any) -> dict[str, Any]:
    """Similarity-only mode: Exact vs Different based on threshold"""
    sim = cosine_sim(embed_sentence(a, cfg), embed_sentence(b, cfg))
    threshold = cfg["engine"].get("sim_threshold", 0.98)
    label = "Exact" if sim > threshold else "Different"
    return {"label": label, "similarity": sim, "evidence": "", "conflict": False}


def annotate_zero_shot(a: str, b: str, cfg: Any) -> dict[str, Any]:
    """Pure zero-shot LLM: one LLM call without similarity hint"""
    prompt = get_prompt_template(cfg)
    from .pipeline import _call_llm

    label, evidence = _call_llm(a, b, prompt, cfg)
    return {"label": label, "similarity": None, "evidence": evidence, "conflict": False}


def annotate_sim_hint(a: str, b: str, cfg: Any) -> dict[str, Any]:
    """LLM call with similarity hint prepended"""
    sim = cosine_sim(embed_sentence(a, cfg), embed_sentence(b, cfg))
    hint = f"Cosine similarity ≈ {sim:.3f}"
    base = get_prompt_template(cfg)
    prompt = f"{hint}\n\n{base}"
    from .pipeline import _call_llm

    label, evidence = _call_llm(a, b, prompt, cfg)
    return {"label": label, "similarity": sim, "evidence": evidence, "conflict": False}


def annotate_vote(a: str, b: str, cfg: Any) -> dict[str, Any]:
    """Vote mode: three LLM calls at different temps with fallback template"""

    def get_vote(temp: float) -> Tuple[str, str]:
        prompt = build_annotation_prompt(a, b, FALLBACK_TEMPLATE)
        # Filter LLM config and set temperature
        client_cfg = {
            k: v for k, v in cfg["llm"].items() if k not in {"top_p", "max_tokens"}
        }
        client_cfg["temperature"] = temp
        client = ArgoGatewayClient(**client_cfg)
        logger.debug(f"Vote LLM call at temp={temp}")
        raw = client.chat(prompt)
        # Try to extract bolded Label — evidence
        match = re.search(r"\*\*([^*—-]+)[—-]\s*([^*]+)\*\*", raw)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        # Fallback simple split
        parts = raw.split("—", 1)
        if len(parts) == 2:
            return parts[0].strip("*- "), parts[1].strip()
        return parts[0].strip("*- "), ""

    try:
        l1, e1 = get_vote(0.8)
        l2, e2 = get_vote(0.2)
        if l1 == l2:
            return {
                "label": l1,
                "similarity": None,
                "evidence": e1 or e2,
                "conflict": False,
            }
        l3, e3 = get_vote(0.0)
        votes = [l1, l2, l3]
        winner = max(set(votes), key=votes.count)
        evidence = {l1: e1, l2: e2, l3: e3}[winner]
        return {
            "label": winner,
            "similarity": None,
            "evidence": evidence,
            "conflict": True,
        }
    except Exception as err:
        logger.error(f"Voting mode failed: {err}")
        return annotate_fallback(a, b, cfg, err)


def annotate_fallback(a: str, b: str, cfg: Any, err: Exception) -> dict[str, Any]:
    """Fallback mode on error: log clearly and do sim-only"""
    logger.error(f"FALLBACK MODE ACTIVATED: {err}")
    base = annotate_local(a, b, cfg)
    base["evidence"] += f"  [FALLBACK—LLM failed: {err}]"
    base["conflict"] = True
    return base
