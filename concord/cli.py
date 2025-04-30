"""
concord.cli
===========
Typer wrapper around the pipeline.

!!  No `str | None` or `Path | None` annotations  !!
"""

from __future__ import annotations

import pathlib as P
import tempfile
import sys
import yaml
import typer
from rich import print as echo

from .pipeline import run_file                          # file workflow

app = typer.Typer(
    add_completion=False,
    help="Concordia – annotation concordance engine",
)


@app.command(name="concord")
def main(                                                # noqa: C901  (CLI only)
    file: str = typer.Argument(
        None, metavar="[FILE]", show_default=False,
        help="Input table (.csv /.tsv /.json)"),
    text_a: str = typer.Option(
        None, help="Free-text annotation A"),
    text_b: str = typer.Option(
        None, help="Free-text annotation B"),
    col_a: str = typer.Option(
        None, help="Column name for annotation A"),
    col_b: str = typer.Option(
        None, help="Column name for annotation B"),
    cfg: str = typer.Option(
        "concord/config.yaml", help="YAML config"),
    mode: str = typer.Option(
        None, help="llm | local | simhint | dual (overrides config)"),
    llm_model: str = typer.Option(
        None, help="Override gateway model – e.g. gpt4o"),
    output: str = typer.Option(
        None, help="Destination CSV"),
    sep: str = typer.Option(
        None, help="Custom delimiter for text files (e.g. '\\t')"),
    force: bool = typer.Option(
        False, "--force",
        help="Overwrite existing OUTPUT instead of resuming/appending"),
):
    # ── sanity ───────────────────────────────────────────────
    if file and (text_a or text_b):
        echo("[red]Give a FILE *or* two strings, not both.[/red]")
        raise typer.Exit(1)
    if not file and not (text_a and text_b):
        echo("[red]Need FILE *or* --text-a + --text-b.[/red]")
        raise typer.Exit(1)

    # ── patch config ─────────────────────────────────────────
    cfg_dict = yaml.safe_load(open(cfg))
    if mode:
        cfg_dict.setdefault("engine", {})["mode"] = mode
    if llm_model:
        cfg_dict.setdefault("llm", {})["model"] = llm_model

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yml") as tmp:
        yaml.safe_dump(cfg_dict, tmp)
        cfg_path = P.Path(tmp.name)

    # ── run ─────────────────────────────────────────────────
    if file:
        out_path = P.Path(output) if output else None
        if out_path and out_path.exists():
            if force:
                out_path.unlink()
                echo(f"[red]✗ removed existing {out_path}[/red]")
            else:
                echo(
                    f"[yellow]⚠ Output {out_path} already exists — "
                    "rerun will *resume* and write only missing rows.\n"
                    "Use --force to overwrite.[/yellow]"
                )

        out = run_file(
            P.Path(file),
            cfg_path,
            col_a, col_b,
            out_path=out_path,
            sep=sep,
        )
        echo(f"[green]✓ wrote {out}[/green]")

    else:
        # ad-hoc pair (import lazily to avoid union signature)
        from .pipeline import run_pair
        label, sim, note = run_pair(text_a, text_b, cfg_path)
        msg = f"label={label}"
        if sim is not None:
            msg += f"   similarity={sim:.3f}"
        if note:
            msg += f"   note={note}"
        echo(msg)


if __name__ == "__main__":
    sys.exit(app())