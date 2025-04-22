import typer, pathlib as P
from .pipeline import run
app=typer.Typer(add_completion=False)
@app.command()
def compare(csv: P.Path, cfg: P.Path="concord/config.yaml", mode: str=None):
    if mode:
        import yaml, shutil, tempfile, os
        cfg_tmp = tempfile.NamedTemporaryFile(delete=False,suffix=".yml")
        d=yaml.safe_load(open(cfg))
        d["engine"]["mode"]=mode
        yaml.safe_dump(d,cfg_tmp.file); cfg_tmp.close(); cfg=cfg_tmp.name
    run(csv, cfg)
    typer.echo(f"✓ wrote {csv.with_suffix('.concordia.csv')}")
if __name__=="__main__": app()
