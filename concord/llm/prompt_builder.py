# concord/llm/prompt_builder.py
import logging

from ..utils import validate_template

logger = logging.getLogger(__name__)

# Hardcoded fallback template for vote mode
FALLBACK_TEMPLATE = """You are a biomedical entity relationship expert.

Your task is to classify the relationship between two biomedical entities:

A: {A}
B: {B}

Classify their relationship using one of the following labels:
- Exact: Entities are identical or functionally equivalent
- Synonym: Different terms for the same concept
- Broader: A is a broader concept than B
- Narrower: A is a narrower concept than B
- Related: Entities are related but don't fit the above categories
- Uninformative: Not enough information to determine relationship
- Different: Entities are completely different concepts

Analyze carefully. Return your answer as: **<Label> — <brief explanation>**"""


def build_annotation_prompt(a: str, b: str, template: str = FALLBACK_TEMPLATE) -> str:
    """
    Fill {A}/{B} placeholders in the given template.
    Raises ValueError if placeholders are missing.
    """
    # Use centralized validation function
    validate_template(template)
    try:
        return template.format(A=a, B=b)
    except Exception as e:
        logger.error(f"Error formatting template: {e}")
        raise
