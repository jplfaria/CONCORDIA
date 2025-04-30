"""
concord.llm.prompts
-------------------
Prompt templates live here.
Task: classify relationship between two gene-function annotations
into the 7-label ontology.
"""
PROMPT_VER = "v1.0"

LABEL_SET = [
    "Exact",
    "Synonym",
    "Broader",
    "Narrower",
    "Related",
    "Uninformative",
    "Different",
]

# ─ few-shot (one per label) ─
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

A: Sigma-70 factor
B: Flagellar motor protein MotA
Different — unrelated functions
"""

_DEFINITIONS = """
Definitions:
• Exact — wording/formatting differs only; identical specific function.
• Synonym — biologically the same function; alternative naming.
• Broader — first description is more general than second (A ⊃ B).
• Narrower — first description is more specific than second (A ⊂ B).
• Related — same pathway / complex / family but not parent–child.
• Uninformative — placeholder or extremely generic description.
• Different — no functional overlap.
"""

SYSTEM_MSG = (
    "You are Concordia, a precise bioinformatics assistant. "
    "Output MUST start with one of these labels: "
    + ", ".join(LABEL_SET)
    + ". Keep the reason under 10 words."
)


# ----------------------------------------------------------------------
def build_annotation_prompt(a: str, b: str, similarity: float | None = None) -> str:
    """
    Build prompt.  If `similarity` provided (simhint mode) append a soft
    prior line, making clear the model may override.
    """
    sim_hint = (
        f"\nCosine similarity ≈ {similarity:.3f} "
        "(weak prior—override if biology disagrees)."
        if similarity is not None
        else ""
    )

    return (
        _FEW_SHOT
        + "\n\n"
        + _DEFINITIONS
        + sim_hint
        + "\n\nClassify the relationship between these two annotations.\n"
        f"A: {a}\nB: {b}\n\n"
        "Respond on **one line** exactly as:\n"
        "<Label> — <very short reason>\n\n"
        f"Allowed labels: {', '.join(LABEL_SET)}."
    )