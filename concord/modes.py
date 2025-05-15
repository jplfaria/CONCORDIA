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
    """Pure zero-shot LLM: single LLM call, optionally prefix similarity hint."""
    from .pipeline import _call_llm

    sim = None
    base_prompt = get_prompt_template(cfg)
    if cfg["engine"].get("sim_hint", False):
        sim = cosine_sim(embed_sentence(a, cfg), embed_sentence(b, cfg))
        hint = f"Cosine similarity ≈ {sim:.3f}"
        prompt = f"{hint}\n\n{base_prompt}"
    else:
        prompt = base_prompt
    label, evidence = _call_llm(a, b, prompt, cfg)
    return {"label": label, "similarity": sim, "evidence": evidence, "conflict": False}


def annotate_vote(a: str, b: str, cfg: Any) -> dict[str, Any]:
    """Vote mode: three LLM calls at different temps with optional similarity hint"""
    # Allow configurable vote temperatures: default [0.8, 0.2, 0.0]
    temps = cfg["engine"].get("vote_temps", [0.8, 0.2, 0.0])
    # Compute similarity hint once if enabled
    sim = None
    if cfg["engine"].get("sim_hint", False):
        sim = cosine_sim(embed_sentence(a, cfg), embed_sentence(b, cfg))

    def get_vote(temp: float) -> Tuple[str, str]:
        base = build_annotation_prompt(a, b, FALLBACK_TEMPLATE)
        if sim is not None:
            hint = f"Cosine similarity ≈ {sim:.3f}"
            prompt = f"{hint}\n\n{base}"
        else:
            prompt = base
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
        l1, e1 = get_vote(temps[0])
        l2, e2 = get_vote(temps[1])
        if l1 == l2:
            return {
                "label": l1,
                "similarity": sim,
                "evidence": e1 or e2,
                "conflict": False,
                "votes": [l1, l2],
            }
        l3, e3 = get_vote(temps[2] if len(temps) > 2 else 0.0)
        votes = [l1, l2, l3]
        winner = max(set(votes), key=votes.count)
        evidence = {l1: e1, l2: e2, l3: e3}[winner]
        return {
            "label": winner,
            "similarity": sim,
            "evidence": evidence,
            "conflict": True,
            "votes": votes,
        }
    except Exception as err:
        logger.error(f"Voting mode failed: {err}")
        return annotate_fallback(a, b, cfg, err)


def annotate_fallback(a: str, b: str, cfg: Any, err: Exception) -> dict[str, Any]:
    """Fallback mode on error: log clearly with error type and do sim-only"""
    err_type = type(err).__name__
    logger.error(f"FALLBACK MODE ACTIVATED: {err_type}: {err}")
    base = annotate_local(a, b, cfg)
    base["evidence"] += f"  [FALLBACK—{err_type}: {err}]"
    base["conflict"] = True
    return base


def annotate_rac(a: str, b: str, cfg: Any) -> dict[str, Any]:
    """Retrieval-augmented classification using similar examples.

    This mode enhances LLM classification by retrieving similar examples
    from the vector store and including them in the prompt to provide
    context for the classification decision.

    Args:
        a: First text to compare
        b: Second text to compare
        cfg: Configuration dictionary

    Returns:
        Dictionary with classification results
    """
    from .embedding import cosine_sim, embed_sentence
    from .pipeline import _call_llm
    from .retrieval import add_classification_example, retrieve_similar_examples

    # Get similarity score if enabled
    sim = None
    if cfg["engine"].get("sim_hint", False):
        sim = cosine_sim(embed_sentence(a, cfg), embed_sentence(b, cfg))

    # Retrieve similar examples
    rac_cfg = cfg.get("rac", {})
    limit = rac_cfg.get("example_limit", 3)
    threshold = rac_cfg.get("similarity_threshold", 0.0)
    examples = retrieve_similar_examples(a, b, cfg, limit=limit, threshold=threshold)

    # Build a prompt with the examples and optional similarity hint
    base_prompt = get_prompt_template(cfg)

    # Add examples section if we have examples
    if examples:
        example_text = (
            "\n\nHere are some similar previously classified pairs for reference:\n\n"
        )
        for i, (ex, score) in enumerate(examples, 1):
            example_text += f"Example {i} (similarity: {score:.3f}):\n"
            example_text += f"A: {ex['text_a']}\n"
            example_text += f"B: {ex['text_b']}\n"
            example_text += f"Label: {ex['label']}\n"
            example_text += f"Reason: {ex['evidence']}\n\n"

        # Insert examples before the final prompt instruction
        parts = base_prompt.rsplit("\n\n", 1)
        if len(parts) == 2:
            prompt = parts[0] + example_text + "\n\n" + parts[1]
        else:
            prompt = base_prompt + example_text
    else:
        prompt = base_prompt

    # Add similarity hint if enabled
    if sim is not None:
        hint = f"Cosine similarity between entities is {sim:.3f}"
        prompt = f"{hint}\n\n{prompt}"

    try:
        # Get LLM response
        label, evidence = _call_llm(a, b, prompt, cfg)

        # Add to vector store if configured for auto-learning
        if rac_cfg.get("auto_store", False) and not cfg.get("test_mode", False):
            metadata = {
                "source": "auto_classified",
                "similarity": sim,
                "examples_used": len(examples),
            }
            add_classification_example(a, b, label, evidence, cfg, metadata)

        return {
            "label": label,
            "similarity": sim,
            "evidence": evidence,
            "conflict": False,
            "examples_used": len(examples),
        }
    except Exception as err:
        logger.error(f"RAC mode failed: {err}")
        return annotate_fallback(a, b, cfg, err)
