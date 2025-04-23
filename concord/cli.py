import typer, pathlib as P, yaml, tempfile
from typing import Optional
from .pipeline import run_file, run_pair

app = typer.Typer(add_completion=False,
                  help="Concordia – genome annotation concordance engine")

# ------------------------------------------------------------------ #
@app.command()
def compare(
    csv: Optional[P.Path] = typer.Argument(
        None, exists=True, readable=True,
        help="CSV with gene_id,old_annotation,new_annotation"),
    text_a: Optional[str] = typer.Option(
        None, help="Free-text annotation A (use with --text-b)"),
    text_b: Optional[str] = typer.Option(
        None, help="Free-text annotation B (use with --text-a)"),
    cfg: P.Path = typer.Option("concord/config.yaml", help="Base YAML config"),
    mode: str = typer.Option(None, help="Mode: llm | local | hybrid"),
    llm_model: str = typer.Option(None, help="Override LLM model name"),
    output: Optional[P.Path] = typer.Option(
        None, help="Write results to this file (CSV mode only)"),
):
    """
    Compare either a CSV file of annotation pairs **or** two ad-hoc strings.
    Default mode is 'llm' (o3-mini on apps-dev).
    """
    # ---- argument sanity -------------------------------------------
    if csv and (text_a or text_b):
        typer.secho("Provide *either* a CSV or two strings, not both.", fg="red")
        raise typer.Exit(1)
    if not csv and not (text_a and text_b):
        typer.secho("Need a CSV file *or* --text-a + --text-b.", fg="red")
        raise typer.Exit(1)

    # ---- patch config on the fly -----------------------------------
    cfg_dict = yaml.safe_load(open(cfg))
    if mode:
        cfg_dict["engine"]["mode"] = mode
    if llm_model:
        cfg_dict["llm"]["model"] = llm_model
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yml") as tmp:
        yaml.safe_dump(cfg_dict, tmp)
        cfg_path = P.Path(tmp.name)

    # ---- run --------------------------------------------------------
    if csv:
        out_file = run_file(csv, cfg_path, out_path=output)
        typer.secho(f"✓ wrote {out_file}", fg="green")
    else:
        label, sim = run_pair(text_a, text_b, cfg_path)
        if sim is None:
            typer.echo(f"label = {label}")
        else:
            typer.echo(f"similarity = {sim:.3f}   label = {label}")

if __name__ == "__main__":
    app()