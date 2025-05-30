"""
concord.cli ─ Typer wrapper
===========================
Command-line interface for the CONCORDIA annotation engine.

v1.2 (2025-05-02)
 • ADD logging options and configuration
 • ADD batch processing controls
 • ADD device selection for embeddings
 • ADD model preloading option
 • IMPROVE error reporting and handling
"""

from __future__ import annotations

import logging
import os
import pathlib as P
import sys
import tempfile
from typing import Optional

import typer
import yaml
from rich import print as echo
from rich.console import Console

from . import setup_logging
from .embedding import clear_cache, preload_model
from .pipeline import run_file, run_pair

# Create console for rich output
console = Console()

app = typer.Typer(
    add_completion=False,
    help="Concordia – annotation concordance engine",
    invoke_without_command=True,
    context_settings={"allow_interspersed_args": True},
)


@app.callback(invoke_without_command=True)
def concord(  # noqa: C901
    file: str = typer.Argument(
        None, metavar="[FILE]", help="Input table (.csv/.tsv/.json)"
    ),
    text_a: str = typer.Option(None, help="Free-text annotation A"),
    text_b: str = typer.Option(None, help="Free-text annotation B"),
    col_a: str = typer.Option(None, help="Column name for annotation A"),
    col_b: str = typer.Option(None, help="Column name for annotation B"),
    cfg: str = typer.Option("config.yaml", help="YAML config at repo root"),
    mode: str = typer.Option(None, help="local | zero-shot | vote"),
    llm_model: str = typer.Option(None, help="Override gateway model"),
    prompt_ver: str = typer.Option(None, help="Freeze a prompt version"),
    output: str = typer.Option(None, help="Destination CSV"),
    overwrite: bool = typer.Option(False, help="Overwrite existing output file"),
    sep: str = typer.Option(None, help="Custom delimiter (e.g. '\\t')"),
    batch_size: int = typer.Option(32, help="Batch size for processing"),
    llm_batch_size: int = typer.Option(
        1, "--llm-batch-size", help="Aggregate N pairs into a single LLM request"
    ),
    llm_stream: Optional[bool] = typer.Option(
        None,
        "--llm-stream/--no-llm-stream",
        help="Force /streamchat (on) or /chat (off). Omit flag for auto mode.",
    ),
    llm_debug: bool = typer.Option(
        False, "--llm-debug", help="Enable verbose Argo Gateway debugging"
    ),
    device: str = typer.Option("cpu", help="Device for embedding model (cpu/cuda)"),
    preload: bool = typer.Option(False, help="Preload embedding model"),
    log_level: str = typer.Option(
        "INFO", help="Logging level (DEBUG/INFO/WARNING/ERROR)"
    ),
    log_file: str = typer.Option(None, help="Log to file in addition to console"),
    list_templates: bool = typer.Option(False, help="List available prompt templates"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed evidence and explanations"
    ),
    sim_hint: bool = typer.Option(
        False, "--sim-hint", help="Prefix similarity hint to LLM prompt"
    ),
):
    """
    Run the CONCORDIA annotation engine on a file or a pair of texts.

    Examples:
      concord data.csv --mode llm
      concord --text-a "Protein A" --text-b "Protein B" --mode duo
    """
    try:
        # Setup logging based on CLI options
        setup_logging(level=log_level.upper(), log_file=log_file)
        logger = logging.getLogger("concord.cli")

        # List available templates if requested
        if list_templates:
            from .llm.prompts import list_available_templates

            templates = list_available_templates()
            console.print("[bold]Available templates:[/bold]")
            for template in templates:
                console.print(f"  • {template}")
            return

        # ── sanity ───────────────────────────────────────────────
        if file and (text_a or text_b):
            echo("[red]Give a FILE *or* two strings – not both.[/red]")
            raise typer.Exit(1)
        if not file and not (text_a and text_b):
            echo("[red]Need FILE *or* --text-a + --text-b.[/red]")
            raise typer.Exit(1)

        # ── patch config (temporary copy) ────────────────────────
        try:
            with open(cfg) as f:
                cfg_dict = yaml.safe_load(f)
        except (IOError, yaml.YAMLError) as e:
            echo(f"[red]Error loading config file: {e}[/red]")
            raise typer.Exit(1)

        # Update config with CLI options
        if mode:
            cfg_dict.setdefault("engine", {})["mode"] = mode
        if llm_model:
            # Normalize model name by removing hyphens to match Argo API requirements
            if llm_model == "gpt-4o":
                llm_model = "gpt4o"
            cfg_dict.setdefault("llm", {})["model"] = llm_model
        if prompt_ver:
            cfg_dict["prompt_ver"] = prompt_ver

        # Add sim_hint flag to engine config
        cfg_dict.setdefault("engine", {})["sim_hint"] = sim_hint

        # Add embedding options
        cfg_dict.setdefault("embedding", {})["device"] = device
        cfg_dict["embedding"]["batch_size"] = batch_size

        # propagate llm_debug to config/environment
        if llm_debug:
            cfg_dict.setdefault("llm", {})["debug"] = True
            # Set env var so ArgoGatewayClient picks it up even if constructed elsewhere
            os.environ["ARGO_DEBUG"] = "1"

        # propagate llm_stream override if provided
        if llm_stream is not None:
            cfg_dict.setdefault("llm", {})["stream"] = llm_stream

        # Write updated config to temporary file
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yml") as tmp:
            yaml.safe_dump(cfg_dict, tmp)
            cfg_path = P.Path(tmp.name)

        # Check Argo Gateway connectivity
        from .llm.argo_gateway import ArgoGatewayClient

        client = ArgoGatewayClient(**cfg_dict.get("llm", {}))
        if client.ping():
            echo("[green]✓ Argo Gateway API reachable[/green]")
        else:
            echo("[yellow]⚠️ Argo Gateway API unreachable; LLM calls may fail[/yellow]")

        # Preload embedding model if requested
        if preload:
            with console.status(
                "[bold green]Preloading embedding model...[/bold green]"
            ):
                preload_model(cfg_dict)
                console.print("[bold green]✓ Model preloaded[/bold green]")

        # ── run ─────────────────────────────────────────────────
        if file:
            # Run on file
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
                echo(f"[green]✓ wrote {out}[/green]")
            except Exception as e:
                logger.error(f"Error processing file: {e}", exc_info=True)
                echo(f"[red]Error processing file: {e}[/red]")
                raise typer.Exit(1)
        else:
            # Run on a single pair
            try:
                with console.status("[bold green]Processing...[/bold green]"):
                    label, sim, note = run_pair(text_a, text_b, cfg_path)

                # Display results
                console.print("[bold]Results:[/bold]")
                console.print(f"  • [bold]Label:[/bold] {label}")
                if sim is not None:
                    console.print(f"  • [bold]Similarity:[/bold] {sim:.3f}")
                if note:
                    if verbose:
                        # Show the full evidence with proper formatting in verbose mode
                        console.print("  • [bold]Evidence:[/bold]")
                        console.print(f"    {note}")
                    else:
                        # In non-verbose mode, show a shortened version if it's too long
                        if len(note) > 50:
                            console.print(
                                f"  • [bold]Evidence:[/bold] {note[:50]}... (use --verbose for full details)"
                            )
                        else:
                            console.print(f"  • [bold]Evidence:[/bold] {note}")
            except Exception as e:
                logger.error(f"Error processing text pair: {e}", exc_info=True)
                echo(f"[red]Error processing text pair: {e}[/red]")
                raise typer.Exit(1)

        # Clean up after run
        clear_cache()  # Free memory from embedding cache

    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        echo(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def validate(
    cfg: str = typer.Option("config.yaml", help="YAML config to validate"),
):
    """Validate a configuration file."""
    try:
        with open(cfg) as f:
            cfg_dict = yaml.safe_load(f)

        # Check for required sections
        errors = []

        if "engine" not in cfg_dict:
            errors.append("Missing 'engine' section")
        elif "mode" not in cfg_dict["engine"]:
            errors.append("Missing 'engine.mode' setting")

        if "llm" not in cfg_dict:
            errors.append("Missing 'llm' section")

        # Report results
        if errors:
            echo("[red]Configuration validation failed:[/red]")
            for error in errors:
                echo(f"  • [red]{error}[/red]")
            raise typer.Exit(1)
        else:
            echo("[green]✓ Configuration is valid[/green]")

    except (IOError, yaml.YAMLError) as e:
        echo(f"[red]Error loading config file: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":  # python -m concord.cli
    sys.exit(app())
