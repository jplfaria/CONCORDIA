"""
concord.llm.prompts
-------------------
All prompt templates live here.

Edit these functions (or add new ones) to experiment with different
instruction styles, few-shot examples, JSON schemas, etc.  The pipeline
code never needs to change.

Current task: classify the relationship between two annotation strings.
"""

# Allowed labels for the classification task
LABEL_SET = ["Identical", "Synonym", "Partial", "New"]

# Optional few-shot examples (keep it short; models with >128k ctx not needed here)
_FEW_SHOT = """\
A: ATP synthase subunit beta
B: ATP synthase β subunit
Identical — wording difference only

A: DNA ligase
B: NAD-dependent DNA ligase
Partial — second string is more specific
"""


def build_annotation_prompt(a: str, b: str) -> str:
    """
    Return a single prompt string instructing the LLM to emit:

        <Label> — <very short reason>

    • <Label> must be exactly one of LABEL_SET
    • Reason should be under ~10 words.
    """
    return (
        f"{_FEW_SHOT}\n\n"
        "Classify the relationship between these two gene/protein "
        "function annotations.\n"
        f"A: {a}\nB: {b}\n\n"
        "Respond on **one line** exactly as:\n"
        "<Label> — <very short reason>\n\n"
        f"Allowed labels: {', '.join(LABEL_SET)}."
    )