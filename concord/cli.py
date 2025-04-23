import typer, pathlib as P, yaml, tempfile, sys
from typing import Optional
from .pipeline import run_file, run_pair

app = typer.Typer(add_completion=False,
                  help="Concordia – annotation concordance engine")

@app.command()
def compare(
    csv: Optional[P.Path] = typer.Argument(
        None, exists=True, readable=True,
        help="CSV file of annotation pairs"),
    text_a: Optional[str] = typer.Option(None, help="Free-text string A"),
    text_b: Optional[str] = typer.Option(None, help="Free-text string B"),
    col_a: Optional[str] = typer.Option(
        None, help="Name of column holding annotation A"),
    col_b: Optional[str] = typer.Option(
        None, help="Name of column holding annotation B"),
    cfg: P.Path = typer.Option("concord/config.yaml", help="YAML config"),
    mode: str = typer.Option(None, help="llm | local | hybrid"),
    llm_model: str = typer.Option(None, help="Override LLM model"),
    output: Optional[P.Path] = typer.Option(None, help="Output CSV path"),
):
    """
    Compare a CSV *or* two ad-hoc strings.
    """
    if csv and (text_a or text_b):
        typer.secho("Provide either a CSV or two strings, not both.", fg="red")
        raise typer.Exit(1)
    if not csv and not (text_a and text_b):
        typer.secho("Need a CSV or --text-a + --text-b.", fg="red")
        raise typer.Exit(1)

    # patch config
    cfg_dict = yaml.safe_load(open(cfg))
    if mode:
        cfg_dict["engine"]["mode"] = mode
    if llm_model:
        cfg_dict["llm"]["model"] = llm_model
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yml") as tmp:
        yaml.safe_dump(cfg_dict, tmp)
        cfg_path = P.Path(tmp.name)

    # run
    if csv:
        out_file = run_file(csv, cfg_path, col_a, col_b, out_path=output)
        typer.secho(f"✓ wrote {out_file}", fg="green")
    else:
        label, sim, note = run_pair(text_a, text_b, cfg_path)
        line = f"label={label}   similarity={sim:.3f}" if sim else f"label={label}"
        if note:
            line += f"   note={note}"
        typer.echo(line)

if __name__ == "__main__":
    app()