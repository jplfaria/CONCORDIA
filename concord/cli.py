import pathlib as P, tempfile, yaml, typer
from typing import Optional

from .pipeline import run_file, run_pair

echo = typer.echo
app = typer.Typer(add_completion=False,
                  help="Concordia – annotation concordance engine (7-label)")

MODE_CHOICES = ["llm", "local", "dual"]

@app.command()
def concord(
    file: Optional[P.Path] = typer.Argument(
        None, exists=True, readable=True, help="CSV/TSV/JSON table of pairs"
    ),
    text_a: Optional[str] = typer.Option(None, help="Free-text string A"),
    text_b: Optional[str] = typer.Option(None, help="Free-text string B"),
    col_a: Optional[str] = typer.Option(None, help="Column with annotation A"),
    col_b: Optional[str] = typer.Option(None, help="Column with annotation B"),
    sep: Optional[str] = typer.Option(None, help="Custom delimiter for text files"),
    cfg: P.Path = typer.Option("concord/config.yaml", help="YAML config"),
    mode: str = typer.Option(None, help="llm | local | dual", show_default=False),
    llm_model: str = typer.Option(None, help="Override LLM model"),
    output: Optional[P.Path] = typer.Option(None, help="Output path"),
):
    """Compare a file *or* two ad-hoc strings."""
    if file and (text_a or text_b):
        echo("[red]Choose either a file or two strings, not both.[/red]")
        raise typer.Exit(1)
    if not file and not (text_a and text_b):
        echo("[red]Need a file or --text-a + --text-b.[/red]")
        raise typer.Exit(1)
    if mode and mode not in MODE_CHOICES:
        echo(f"[red]Mode must be one of {MODE_CHOICES}.[/red]")
        raise typer.Exit(1)

    cfg_dict = yaml.safe_load(open(cfg))
    if mode:
        cfg_dict.setdefault("engine", {})["mode"] = mode
    if llm_model:
        cfg_dict.setdefault("llm", {})["model"] = llm_model
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yml") as tmp:
        yaml.safe_dump(cfg_dict, tmp)
        cfg_path = P.Path(tmp.name)

    if file:
        out = run_file(file, cfg_path, col_a, col_b, out_path=output, sep=sep)
        echo(f"[green]✓ wrote {out}[/green]")
    else:
        label, sim, note = run_pair(text_a, text_b, cfg_path)
        bits = [f"label={label}"]
        if sim is not None:
            bits.append(f"{_SIM_COL}={sim:.3f}")
        if note:
            bits.append(f"note={note}")
        echo("   ".join(bits))


if __name__ == "__main__":
    app()