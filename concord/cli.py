"""
concord.cli ─ Streamlined CLI
=============================
Command-line interface for the CONCORDIA annotation engine.

v1.3 (2025-05-02)
 • SIMPLIFIED argument handling and config patching
 • GROUPED related CLI options
 • STREAMLINED logic flow
"""

from __future__ import annotations

import logging
import os
import pathlib as P
import tempfile
from typing import Optional

import typer
import yaml
from rich import print as echo
from rich.console import Console

from . import setup_logging
from .embedding import clear_cache, preload_model
from .pipeline import run_file, run_pair
from .utils import load_yaml_config

# Create console for rich output
console = Console()

app = typer.Typer(
    add_completion=False,
    help="Concordia – annotation concordance engine",
    invoke_without_command=True,
    context_settings={"allow_interspersed_args": True},
)


def _update_config(cfg_dict: dict, **options) -> None:
    """Update configuration with CLI options efficiently."""
    # Engine options
    if options["mode"]:
        cfg_dict.setdefault("engine", {})["mode"] = options["mode"]
    cfg_dict.setdefault("engine", {})["sim_hint"] = options["sim_hint"]

    # LLM options
    if options["llm_model"]:
        # Normalize model names
        model = options["llm_model"].replace("-", "")  # gpt-4o -> gpt4o
        cfg_dict.setdefault("llm", {})["model"] = model
    if options["llm_debug"]:
        cfg_dict.setdefault("llm", {})["debug"] = True
        os.environ["ARGO_DEBUG"] = "1"
    if options["llm_stream"] is not None:
        cfg_dict.setdefault("llm", {})["stream"] = options["llm_stream"]

    # Prompt options
    if options["prompt_ver"]:
        cfg_dict["prompt_ver"] = options["prompt_ver"]

    # Embedding options
    cfg_dict.setdefault("embedding", {}).update(
        {"device": options["device"], "batch_size": options["batch_size"]}
    )


@app.callback(invoke_without_command=True)
def concord(
    # Input options
    file: str = typer.Argument(None, metavar="[FILE]", help="Input CSV/TSV file"),
    text_a: str = typer.Option(None, help="First annotation text"),
    text_b: str = typer.Option(None, help="Second annotation text"),
    col_a: str = typer.Option(None, help="Column name for first annotation"),
    col_b: str = typer.Option(None, help="Column name for second annotation"),
    # Configuration options
    cfg: str = typer.Option("config.yaml", help="YAML config file"),
    mode: str = typer.Option(None, help="Annotation mode: local|zero-shot|vote|rac"),
    prompt_ver: str = typer.Option(None, help="Prompt template version"),
    # LLM options
    llm_model: str = typer.Option(None, help="LLM model override"),
    llm_batch_size: int = typer.Option(1, help="Batch size for LLM calls"),
    llm_stream: Optional[bool] = typer.Option(
        None, "--llm-stream/--no-llm-stream", help="Force streaming mode"
    ),
    llm_debug: bool = typer.Option(False, "--llm-debug", help="Enable LLM debugging"),
    # Processing options
    batch_size: int = typer.Option(32, help="Processing batch size"),
    device: str = typer.Option("cpu", help="Device for embeddings (cpu/cuda)"),
    sim_hint: bool = typer.Option(False, "--sim-hint", help="Include similarity hints"),
    preload: bool = typer.Option(False, help="Preload embedding model"),
    # Output options
    output: str = typer.Option(None, help="Output CSV file path"),
    overwrite: bool = typer.Option(False, help="Overwrite existing output"),
    sep: str = typer.Option(None, help="CSV delimiter"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output"),
    # Utility options
    log_level: str = typer.Option("INFO", help="Logging level"),
    log_file: str = typer.Option(None, help="Log file path"),
    list_templates: bool = typer.Option(False, help="List available templates"),
):
    """
    Run CONCORDIA annotation engine on files or text pairs.

    Examples:
      concord data.csv --mode zero-shot
      concord --text-a "Protein A" --text-b "Protein B"
    """
    try:
        # Setup logging
        setup_logging(level=log_level.upper(), log_file=log_file)
        logger = logging.getLogger("concord.cli")

        # Handle utility commands
        if list_templates:
            from .llm.prompts import list_available_templates

            templates = list_available_templates()
            console.print("[bold]Available templates:[/bold]")
            for template in templates:
                console.print(f"  • {template}")
            return

        # Validate input arguments
        if file and (text_a or text_b):
            echo("[red]Specify either a FILE or text pair, not both.[/red]")
            raise typer.Exit(1)
        if not file and not (text_a and text_b):
            echo("[red]Provide either a FILE or both --text-a and --text-b.[/red]")
            raise typer.Exit(1)

        # Load and update configuration
        try:
            cfg_dict = load_yaml_config(cfg)
        except ValueError as e:
            echo(f"[red]Config error: {e}[/red]")
            raise typer.Exit(1)

        # Update config with CLI options
        # Exclude cfg_dict from locals to avoid passing it twice
        options = {k: v for k, v in locals().items() if k != "cfg_dict"}
        _update_config(cfg_dict, **options)

        # Create temporary config file
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yml") as tmp:
            yaml.safe_dump(cfg_dict, tmp)
            cfg_path = P.Path(tmp.name)

        # Optional: Check connectivity and preload model (skip for single pairs)
        if not cfg_dict.get("skip_connectivity_check", False) and file:
            from .llm.argo_gateway import ArgoGatewayClient

            try:
                client = ArgoGatewayClient(**cfg_dict.get("llm", {}))
                if client.ping():
                    echo("[green]✓ Gateway reachable[/green]")
                else:
                    echo("[yellow]⚠️ Gateway unreachable[/yellow]")
            except Exception:
                echo("[yellow]⚠️ Could not check gateway[/yellow]")

        if preload:
            with console.status("[bold green]Loading model...[/bold green]"):
                preload_model(cfg_dict)
                console.print("[bold green]✓ Model loaded[/bold green]")

        # Execute main logic
        if file:
            _process_file(
                file,
                cfg_path,
                col_a,
                col_b,
                output,
                overwrite,
                sep,
                batch_size,
                llm_batch_size,
                logger,
            )
        else:
            _process_pair(text_a, text_b, cfg_path, verbose)

        # Cleanup
        clear_cache()

    except Exception as e:
        logger.error(f"CLI error: {e}", exc_info=True)
        echo(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


def _process_file(
    file: str,
    cfg_path: P.Path,
    col_a: str,
    col_b: str,
    output: str,
    overwrite: bool,
    sep: str,
    batch_size: int,
    llm_batch_size: int,
    logger,
) -> None:
    """Process file input."""
    try:
        out = run_file(
            P.Path(file),
            cfg_path,
            col_a,
            col_b,
            out_path=P.Path(output) if output else None,
            overwrite=overwrite,
            sep=sep,
            batch_size=batch_size,
            llm_batch_size=llm_batch_size,
        )
        echo(f"[green]✓ Output written to {out}[/green]")
    except Exception as e:
        logger.error(f"File processing failed: {e}")
        echo(f"[red]File processing failed: {e}[/red]")
        raise typer.Exit(1)


def _process_pair(text_a: str, text_b: str, cfg_path: P.Path, verbose: bool) -> None:
    """Process text pair input."""
    try:
        with console.status("[bold green]Processing...[/bold green]"):
            label, sim, note = run_pair(text_a, text_b, cfg_path)

        # Display results
        console.print(f"[bold]Label:[/bold] {label}")
        if sim is not None:
            console.print(f"[bold]Similarity:[/bold] {sim:.3f}")
        if note:
            if verbose or len(note) <= 50:
                console.print(f"[bold]Evidence:[/bold] {note}")
            else:
                console.print(
                    f"[bold]Evidence:[/bold] {note[:50]}... (use -v for full)"
                )
    except Exception as e:
        echo(f"[red]Text processing failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def validate(cfg: str = typer.Option("config.yaml", help="Config file to validate")):
    """Validate a configuration file."""
    try:
        from .utils import validate_config

        cfg_dict = load_yaml_config(cfg)
        errors = validate_config(cfg_dict)

        if errors:
            echo("[red]Validation failed:[/red]")
            for error in errors:
                echo(f"  • [red]{error}[/red]")
            raise typer.Exit(1)
        else:
            echo("[green]✓ Configuration valid[/green]")
    except Exception as e:
        echo(f"[red]Validation error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
