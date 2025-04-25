"""
concord.cli
===========
Thin Typer wrapper around pipeline.run_file / run_pair
"""

from __future__ import annotations
import pathlib as P, tempfile, yaml, typer
from typing import Optional

from rich import print as echo          # nice colours

from .pipeline import run_file, run_pair

app = typer.Typer(add_completion=False,
                  help="Concordia – annotation concordance engine")


@app.command()
def concord(                                            # noqa: C901  (CLI only)
    file: Optional[str] = typer.Argument(               # table path (.csv/.tsv/.json)
        None, help="Input table with two annotation columns"),
    text_a: Optional[str] = typer.Option(
        None, help="Free-text annotation A"),
    text_b: Optional[str] = typer.Option(
        None, help="Free-text annotation B"),
    col_a: Optional[str] = typer.Option(
        None, help="Name of column holding annotation A"),
    col_b: Optional[str] = typer.Option(
        None, help="Name of column holding annotation B"),
    cfg: str = typer.Option(
        "concord/config.yaml", help="Path to YAML config"),
    mode: str = typer.Option(
        None, help="llm | local | dual  (overrides config)"),
    llm_model: str = typer.Option(
        None, help="Override gateway model – e.g. gpt4o"),
    output: Optional[str] = typer.Option(
        None, help="Destination CSV path"),
    sep: Optional[str] = typer.Option(
        None, help="Custom delimiter for text files (e.g. '\\t')"),
):
    """
    • Give a FILE to process a whole table  **or**
    • Give --text-a + --text-b to compare a single pair.
    """
    # ------------------------------------------------------------------ sanity
    if file and (text_a or text_b):
        echo("[red]Provide either FILE *or* --text-a/--text-b, not both.[/red]")
        raise typer.Exit(1)
    if not file and not (text_a and text_b):
        echo("[red]Need FILE *or* --text-a + --text-b.[/red]")
        raise typer.Exit(1)

    # ------------------------------------------------------------------ config
    cfg_dict = yaml.safe_load(open(cfg))
    if mode:
        cfg_dict.setdefault("engine", {})["mode"] = mode
    if llm_model:
        cfg_dict.setdefault("llm", {})["model"] = llm_model

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yml") as tmp:
        yaml.safe_dump(cfg_dict, tmp)
        cfg_path = P.Path(tmp.name)

    # ------------------------------------------------------------------ run
    if file:
        out = run_file(
            P.Path(file),
            cfg_path,
            col_a, col_b,
            out_path=P.Path(output) if output else None,
            sep=sep,
        )
        echo(f"[green]✓ wrote {out}[/green]")

    else:  # ad-hoc
        label, sim, note = run_pair(text_a, text_b, cfg_path)
        msg = f"label={label}"
        if sim is not None:
            msg += f"   similarity={sim:.3f}"
        if note:
            msg += f"   note={note}"
        echo(msg)


if __name__ == "__main__":   # python -m concord.cli
    app()