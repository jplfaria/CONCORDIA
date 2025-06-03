"""
concord.llm.prompts
===================
Central registry for prompt templates.

v2.0 (2025-05-02)
 • REMOVE bucket routing and v1.1-* templates (deprecated)
 • FIX 6-class ontology documentation
 • SIMPLIFY template management
 • ADD support for external template files
 • ADD comprehensive error handling

* PROMPT_VER           – default template version
* _TEMPLATES           – version-tag ➜ template content
* LABEL_SET            – 6-class ontology (validation & docs)
* get_prompt_template  – returns template content
* build_annotation_prompt() – fills {A}/{B} placeholders
"""

from __future__ import annotations

import logging
import pathlib as P
from typing import Any, Dict, List

from ..utils import validate_template

# Configure logging
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# 6-class ontology
# -------------------------------------------------------------------
LABEL_SET: set[str] = {
    "Exact",
    "Synonym",
    "Broader",
    "Narrower",
    "Uninformative",
    "Different",
}

# -------------------------------------------------------------------
# Default template version
# -------------------------------------------------------------------
PROMPT_VER = "v3.2"  # Default to our current best template

# -------------------------------------------------------------------
# Template registry
# -------------------------------------------------------------------
_TEMPLATES: dict[str, str] = {
    # Basic fallback template
    "v1.0": (
        "A: {A}\n"
        "B: {B}\n"
        "### Task\n"
        "Return **<Label> — <very short reason>**.\n"
        "Allowed labels: " + ", ".join(sorted(LABEL_SET))
    ),
    # External templates (loaded from files)
    "v3.2": "",  # Content loaded from file
    "v3.2-CoT": "",  # Content loaded from file
}

# -------------------------------------------------------------------
# External template directory
# -------------------------------------------------------------------
_TEMPLATE_DIR = P.Path(__file__).parent / "templates"


def _load_external_templates() -> None:
    """Load templates from external files."""
    global _TEMPLATES

    if not _TEMPLATE_DIR.exists():
        logger.info(f"Creating template directory {_TEMPLATE_DIR}")
        _TEMPLATE_DIR.mkdir(exist_ok=True, parents=True)
        _create_default_template()
        return

    try:
        template_files = list(_TEMPLATE_DIR.glob("template_*.txt"))
        if not template_files:
            logger.warning("No template files found. Creating default template.")
            _create_default_template()
            return

        for file_path in template_files:
            try:
                version = file_path.stem.replace("template_", "")
                logger.debug(
                    f"Processing template file: {file_path.name} (version: {version})"
                )
                # Skip deprecated bucket templates and test templates
                if (
                    version.startswith("v1.1-")
                    or version.startswith("v1.2-")
                    or version.startswith("v1.3-")
                    or "test" in version.lower()
                    or "bug" in version.lower()
                ):
                    logger.info(f"Skipping deprecated template: {version}")
                    continue
                with open(file_path, "r") as f:
                    content = f.read().strip()
                if not validate_template(content, raise_error=False):
                    logger.warning(
                        f"Invalid template '{version}'. Skipping. Content preview: {content[:60]}"
                    )
                    continue
                _TEMPLATES[version] = content
                logger.info(f"Loaded template: {version}")
            except Exception as e:
                logger.error(f"Error loading template from {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error loading external templates: {e}")


def _create_default_template() -> None:
    """Create default template file if none exists."""
    default_template_content = """You are an expert curator of gene functional annotations (SwissProt / RAST style) for microbial genomes.
Your task is to classify the semantic relationship between two independent protein annotations.
Follow community ontology practice (EC, UniProt, KEGG).
If either annotation involves phage structural proteins, also apply the phage-specific heuristics below.

Annotation A: {A}
Annotation B: {B}

**Allowed labels – choose exactly one**

• **Exact** – wording differences only (capitalisation, punctuation, accepted synonyms).
• **Synonym** – same biological entity but nomenclature **or EC renumbering** changes.
• **Broader** – **B** is more generic; loses specificity present in **A**.
• **Narrower** – **B** adds specificity (extra EC digits, substrate, sub-unit, phage part).
• **Different** – unrelated proteins or functions (includes pathway neighbours or fused variants).
• **Uninformative** – neither term provides enough functional information.

**Key heuristics – apply in order; stop at first match**

0. **Insufficient information**
   • If *both* annotations are generic ("hypothetical", "putative protein", locus tag only, "gp#", etc.) and give no catalytic/structural detail ⇒ *Uninformative*.

1. **EC numbers**
   • Added digits ⇒ *Narrower*. • Modern renumbering (3.x → 7.x, 5.99 → 5.6) ⇒ *Synonym*.
   • EC jump that changes catalytic class ⇒ *Different*.

2. **Qualifier clean-up**
   • Removing only "hypothetical", "putative", "protein", gene symbol, or adding "protein" ⇒ *Exact*.
   • Qualifier dropped **and** new specific function appears ⇒ *Narrower*.

3. **Phage-specific rules** (structural proteins only)
   • "gp#", locus tags, "FIGxxxxx" resolved to named virion part ⇒ *Narrower*.
   • Generic "Phage protein" ⇒ named virion part ⇒ *Narrower*; reverse ⇒ *Broader*.

4. **Sub-unit swaps / distinct chains**
   • Alpha↔Beta, large↔small, different terminase subunits, etc. ⇒ *Different* (unless pure renaming).

5. **Frameshift / truncated / hash comments**
   • "# frameshift", "## RluD", "(fragment)" do not change biology ⇒ *Exact* unless explicit loss-of-function.

Respond **in exactly this format** (no extra text):

<Label> — <short justification referencing the rule(s) or evidence>
"""
    try:
        _TEMPLATE_DIR.mkdir(exist_ok=True, parents=True)
        file_path = _TEMPLATE_DIR / f"template_{PROMPT_VER}.txt"
        if not file_path.exists():
            with open(file_path, "w") as f:
                f.write(default_template_content)
            logger.info(f"Created default template: {file_path}")
    except Exception as e:
        logger.error(f"Failed to create default template: {e}")


# Load external templates when module is imported
try:
    _load_external_templates()
except Exception as e:
    logger.error(f"Failed to load external templates: {e}")


# -------------------------------------------------------------------
# Template management functions
# -------------------------------------------------------------------
def get_prompt_template(
    cfg: Dict[str, Any],
    *,
    ver: str | None = None,
    bucket_pair: (
        tuple[str, str] | None
    ) = None,  # Kept for backward compatibility but ignored
) -> str:
    """Return the appropriate prompt template content."""
    try:
        # Determine template version with clear priority
        if ver:
            key = ver
        elif "prompt_ver" in cfg:
            key = cfg["prompt_ver"]
        else:
            key = PROMPT_VER

        # Get template content
        if key in _TEMPLATES:
            template = _TEMPLATES[key]
            if template:  # Non-empty template
                validate_template(template)
                return template
            else:
                raise ValueError(f"Template '{key}' is empty (file not loaded)")

        raise ValueError(f"Template version '{key}' not found")
    except Exception as e:
        logger.error(f"Error getting prompt template: {e}")
        raise


def build_annotation_prompt(a: str, b: str, template: str) -> str:
    """Fill {A}/{B} placeholders in template with validation."""
    try:
        if "{A}" not in template or "{B}" not in template:
            raise ValueError("Template missing required {A} or {B} placeholders")
        return template.format(A=a, B=b)
    except Exception as e:
        logger.error(f"Error building annotation prompt: {e}")
        raise ValueError(f"Failed to build annotation prompt: {e}") from e


def list_available_templates() -> List[str]:
    """Return sorted list of available template versions."""
    return sorted([k for k, v in _TEMPLATES.items() if v])  # Only non-empty templates


def save_template(version: str, content: str) -> bool:
    """Save template to file with validation."""
    if not validate_template(content, raise_error=False):
        logger.warning(f"Cannot save invalid template: {version}")
        return False

    try:
        _TEMPLATE_DIR.mkdir(exist_ok=True, parents=True)
        file_path = _TEMPLATE_DIR / f"template_{version}.txt"

        with open(file_path, "w") as f:
            f.write(content)

        _TEMPLATES[version] = content
        logger.info(f"Saved template: {version}")
        return True
    except Exception as e:
        logger.error(f"Failed to save template {version}: {e}")
        return False
