# concord/modes.py
import logging
from typing import Any, Dict, Tuple

from .embedding import cosine_sim, embed_sentence
from .llm.argo_gateway import ArgoGatewayClient, _parse
from .llm.prompts import build_annotation_prompt, get_prompt_template

logger = logging.getLogger(__name__)


def annotate_local(a: str, b: str, cfg: Any) -> Dict[str, Any]:
    """Similarity-only mode: Exact vs Different based on threshold"""
    sim = cosine_sim(embed_sentence(a, cfg), embed_sentence(b, cfg))
    threshold = cfg["engine"].get("sim_threshold", 0.98)
    label = "Exact" if sim > threshold else "Different"
    return {"label": label, "similarity": sim, "evidence": "", "conflict": False}


def annotate_zero_shot(
    a: str, b: str, cfg: Any, client: ArgoGatewayClient = None
) -> Dict[str, Any]:
    """Single LLM call with optional similarity hint and client reuse."""
    sim = None
    template = get_prompt_template(cfg)

    if cfg["engine"].get("sim_hint", False):
        sim = cosine_sim(embed_sentence(a, cfg), embed_sentence(b, cfg))
        hint = f"Cosine similarity ≈ {sim:.3f}\n\n"
        template = hint + template

    # Create client only if not provided
    if client is None:
        llm_cfg = {
            k: v
            for k, v in cfg["llm"].items()
            if k not in {"temperature", "top_p", "max_tokens"}
        }
        client = ArgoGatewayClient(**llm_cfg)

    prompt = build_annotation_prompt(a, b, template)

    try:
        raw = client.chat(prompt)
        label, evidence = _parse(raw)
        return {
            "label": label,
            "similarity": sim,
            "evidence": evidence,
            "conflict": False,
        }
    except Exception as e:
        logger.error(f"Zero-shot annotation failed: {e}")
        return annotate_fallback(a, b, cfg, e)


def annotate_vote(
    a: str, b: str, cfg: Any, client: ArgoGatewayClient = None
) -> Dict[str, Any]:
    """Vote mode: three LLM calls at different temperatures with client reuse."""
    temps = cfg["engine"].get("vote_temps", [0.8, 0.2, 0.0])
    sim = None

    if cfg["engine"].get("sim_hint", False):
        sim = cosine_sim(embed_sentence(a, cfg), embed_sentence(b, cfg))

    # Create base client once if not provided
    if client is None:
        llm_cfg = {
            k: v for k, v in cfg["llm"].items() if k not in {"top_p", "max_tokens"}
        }
        client = ArgoGatewayClient(**llm_cfg)

    def get_vote(temp: float) -> Tuple[str, str]:
        template = get_prompt_template(cfg)
        if sim is not None:
            template = f"Cosine similarity ≈ {sim:.3f}\n\n" + template

        # Use provided client but update temperature
        # Note: We can't change temperature on existing client,
        # but for vote mode we need different temps per call
        if temp != 0.0:  # Only create new client if temp is different
            llm_cfg = {
                k: v for k, v in cfg["llm"].items() if k not in {"top_p", "max_tokens"}
            }
            llm_cfg["temperature"] = temp
            temp_client = ArgoGatewayClient(**llm_cfg)
        else:
            temp_client = client

        prompt = build_annotation_prompt(a, b, template)
        raw = temp_client.chat(prompt)
        return _parse(raw)

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
        evidence = next(
            (e for label, e in [(l1, e1), (l2, e2), (l3, e3)] if label == winner), ""
        )

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


def annotate_fallback(a: str, b: str, cfg: Any, err: Exception) -> Dict[str, Any]:
    """Fallback mode: log error and use similarity-only classification."""
    logger.error(f"FALLBACK MODE: {type(err).__name__}: {err}")
    result = annotate_local(a, b, cfg)
    result["evidence"] += f" [FALLBACK—{type(err).__name__}]"
    result["conflict"] = True
    return result


def annotate_rac(
    a: str, b: str, cfg: Any, client: ArgoGatewayClient = None
) -> Dict[str, Any]:
    """Retrieval-augmented classification using similar examples with client reuse.

    This mode enhances LLM classification by retrieving similar examples
    from the vector store and including them in the prompt to provide
    context for the classification decision.

    Args:
        a: First text to compare
        b: Second text to compare
        cfg: Configuration dictionary
        client: Optional ArgoGatewayClient to reuse

    Returns:
        Dictionary with classification results
    """
    from .retrieval import add_classification_example, retrieve_similar_examples

    sim = None
    if cfg["engine"].get("sim_hint", False):
        sim = cosine_sim(embed_sentence(a, cfg), embed_sentence(b, cfg))

    # Retrieve examples
    rac_cfg = cfg.get("rac", {})
    limit = rac_cfg.get("example_limit", 3)
    threshold = rac_cfg.get("similarity_threshold", 0.0)
    examples = retrieve_similar_examples(a, b, cfg, limit=limit, threshold=threshold)

    # Build prompt
    template = get_prompt_template(cfg)

    if examples:
        example_text = "\n\nSimilar examples:\n\n"
        for i, (ex, score) in enumerate(examples, 1):
            example_text += (
                f"{i}. A: {ex['text_a']} | B: {ex['text_b']} | Label: {ex['label']}\n"
            )

        template = template + example_text

    if sim is not None:
        template = f"Cosine similarity: {sim:.3f}\n\n" + template

    try:
        # Create client only if not provided
        if client is None:
            llm_cfg = {
                k: v
                for k, v in cfg["llm"].items()
                if k not in {"temperature", "top_p", "max_tokens"}
            }
            client = ArgoGatewayClient(**llm_cfg)

        prompt = build_annotation_prompt(a, b, template)
        raw = client.chat(prompt)
        label, evidence = _parse(raw)

        # Auto-store if configured
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
