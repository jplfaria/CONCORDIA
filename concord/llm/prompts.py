"""
concord.llm.prompts
-------------------
Prompt templates live here.
Task: classify the relationship between two gene-function annotations.
"""

PROMPT_VER = "v2025-04-24"          # <= 🆕  keep a changelog-friendly tag

LABEL_SET = [
    "Exact",
    "Synonym",
    "Broader",
    "Narrower",
    "Related",
    "Uninformative",
    "Different",
]

# ----------------------------------------------------------------------
_FEW_SHOT = """\
A: ATP synthase subunit beta
B: ATP synthase β subunit
Exact — wording difference only

A: DNA ligase
B: NAD-dependent DNA ligase
Narrower — second is more specific

A: RecA protein
B: DNA recombinase A
Synonym — alternative name

A: ABC transporter
B: ABC transporter, maltose specific
Broader — first is more general

A: DNA gyrase A subunit
B: DNA topoisomerase IV subunit A
Related — same pathway but not parent–child

A: Hypothetical protein
B: Hypothetical protein
Uninformative — placeholder

A: RNA polymerase sigma-70 factor
B: Flagellar motor protein MotA
Different — unrelated functions
"""

_DEFINITIONS = """
Definitions:
• Exact — wording/formatting differs only, identical specific function.
• Synonym — biologically the same function, alternative naming.
• Broader — first description is more general than second (A ⊃ B).
• Narrower — first description is more specific than second (A ⊂ B).
• Related — same pathway / complex / family but not parent–child.
• Uninformative — placeholder or extremely generic description.
• Different — no functional overlap.
"""


def build_annotation_prompt(a: str, b: str) -> str:
    """Return one prompt string → LLM must emit '<Label> — <reason>'."""
    return (
        f"{_FEW_SHOT}\n\n{_DEFINITIONS}\n\n"
        "Classify the relationship between these two gene/protein "
        "function annotations.\n"
        f"A: {a}\nB: {b}\n\n"
        "Respond on **one line** exactly as:\n"
        "<Label> — <very short reason>\n\n"
        f"Allowed labels: {', '.join(LABEL_SET)}."
    )