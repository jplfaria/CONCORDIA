import logging
from typing import Any, Dict, List, Optional, Tuple

from .prompts import _TEMPLATES
from .prompts import get_prompt_template as _old_get

logger = logging.getLogger(__name__)


def list_available_templates() -> List[str]:
    """Return sorted list of available template versions."""
    return sorted(_TEMPLATES.keys())


def get_prompt_template(
    cfg: Dict[str, Any],
    ver: Optional[str] = None,
    bucket_pair: Optional[Tuple[str, str]] = None,
) -> str:
    """
    Return the appropriate prompt template, delegating to original logic.
    Args:
        cfg: Configuration dictionary
        ver: Explicit version override
        bucket_pair: Pair of texts for bucket routing
    """
    return _old_get(cfg, ver=ver, bucket_pair=bucket_pair)
