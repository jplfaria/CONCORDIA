# concord/cli.py
import typer, pathlib as P, yaml, tempfile
from typing import Optional
from .pipeline import run

app = typer.Typer(add_completion=False)

@app.command()
def main(
    csv: P.Path,
    cfg: P.Path = "concord/config.yaml",
    mode: Optional[str] = typer.Option(None, help="local | llm | hybrid"),
    llm_model: Optional[str] = typer.Option(None, help="override LLM model")
):
    cfg_dict = yaml.safe_load(open(cfg))

    if mode:
        cfg_dict["engine"]["mode"] = mode
    if llm_model:
        cfg_dict["llm"]["model"] = llm_model

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as tmp:
        yaml.safe_dump(cfg_dict, tmp)
        tmp_path = tmp.name

    run(csv, tmp_path)
    typer.echo(f"✓ wrote {csv.with_suffix('.concordia.csv')}")

if __name__ == "__main__":
    app()