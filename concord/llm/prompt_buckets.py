"""
concord.llm.prompt_buckets
==========================
Few-shot helper for the “bucket” prompt variants (v1.1-*).

Only responsibility: append up to N illustrative examples to the
bucket header string defined in prompts.py.  Keeping it isolated means
you can grow / tune the corpus without touching core logic.

Author: Concordia dev team – 2025-05-01
"""

from __future__ import annotations

from textwrap import dedent

# -------------------------------------------------------------------
# Few-shot corpora  (extend or tweak freely)
# -------------------------------------------------------------------

EXAMPLES: dict[str, list[tuple[str, str, str]]] = {
    # --------------  bacteriophage bucket  -------------------------
    "v1.1-phage": [
        (
            "Phage major capsid protein",
            "Phage major capsid protein",
            "Exact — identical names",
        ),
        (
            "Phage tail fiber",
            "Phage tail fiber protein",
            "Synonym — same function + “protein” suffix",
        ),
        (
            "Phage portal protein",
            "Hypothetical protein",
            "Uninformative — generic second term",
        ),
    ],
    # --------------  enzyme bucket  -------------------------------
    "v1.1-enzyme": [
        (
            "Beta-galactosidase (EC 3.2.1.23)",
            "LacZ beta-galactosidase",
            "Synonym — same enzyme",
        ),
        (
            "Alcohol dehydrogenase (EC 1.1.1.1)",
            "Short-chain dehydrogenase",
            "Broader — B is super-family",
        ),
        (
            "Glucose-6-phosphate isomerase",
            "Phosphoglucose isomerase",
            "Exact — alternative name",
        ),
    ],
    # --------------  general bucket  ------------------------------
    "v1.1-general": [
        ("DNA repair protein RecA", "Recombinase A", "Synonym — same protein"),
        (
            "Hypothetical protein",
            "Uncharacterised protein",
            "Uninformative — both generic",
        ),
        ("DNA ligase", "RNA polymerase", "Different — distinct functions"),
    ],
}

# -------------------------------------------------------------------
# Public factory
# -------------------------------------------------------------------


class BucketPrompt:
    """
    Build a prompt string for a given bucket id, including ≤ N examples.
    """

    @classmethod
    def build(cls, key: str, *, n: int = 3) -> str:
        """
        Parameters
        ----------
        key : str
            ‘v1.1-phage’ | ‘v1.1-enzyme’ | ‘v1.1-general’
        n : int, default 3
            Number of few-shot examples to prepend (max).

        Returns
        -------
        str
            Prompt template ready for `.format(A=…, B=…)`.
        """
        from .prompts import _TEMPLATES, LABEL_SET  # late import avoids cycle

        if key not in _TEMPLATES:
            raise ValueError(f"Unknown bucket header '{key}'")
        if key not in EXAMPLES:
            raise ValueError(f"No examples registered for '{key}'")

        header = _TEMPLATES[key]
        shots = EXAMPLES[key][:n]

        joined_examples = []
        for a, b, lbl in shots:
            joined_examples.append(f"A: {a}\nB: {b}\n{lbl}")

        examples_block = "\n\n### Examples\n" + "\n\n".join(joined_examples)
        legend = "\n\nAllowed labels: " + ", ".join(sorted(LABEL_SET))

        return dedent(f"""{header}{examples_block}{legend}""")
