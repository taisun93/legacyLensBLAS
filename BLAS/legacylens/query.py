"""
LegacyLens CLI — query BLAS codebase in natural language.
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from retriever import get_full_file, get_collection
from query_pipeline import run_query, format_timing

console = Console()

SNIPPET_MAX = 800


def _score_color(score: float) -> str:
    if score >= 0.8:
        return "green"
    if score >= 0.6:
        return "yellow"
    return "red"


def _format_snippet(code: str) -> Syntax:
    """Syntax-highlight Fortran snippet, truncated."""
    truncated = code[:SNIPPET_MAX] + ("..." if len(code) > SNIPPET_MAX else "")
    return Syntax(truncated, "fortran", theme="monokai", line_numbers=False)


def run_query_cli(query: str, k: int = 5, feature: str | None = None) -> list[dict]:
    """Execute search and generation, return results + answer."""
    results, answer, timing = run_query(query, k=k, feature=feature)

    # Display results
    table = Table(title="Top Results", show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", width=6)
    table.add_column("File", width=20)
    table.add_column("Lines", width=12)
    table.add_column("Routine", width=12)
    table.add_column("Precision", width=10)
    table.add_column("Operation", width=25)

    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        score = r.get("similarity", 0)
        score_color = _score_color(score)
        table.add_row(
            str(i),
            Text(f"{score:.2f}", style=score_color),
            meta.get("file_name", "—"),
            f"{meta.get('start_line', '?')}-{meta.get('end_line', '?')}",
            meta.get("routine_name", "—") or "—",
            meta.get("precision", "—"),
            meta.get("operation_type", "—"),
        )
    console.print(table)

    # Code snippets
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        score_color = _score_color(r.get("similarity", 0))
        header = f"Result {i}: {meta.get('file_name', '?')} lines {meta.get('start_line')}-{meta.get('end_line')}"
        if meta.get("routine_name"):
            header += f" — {meta['routine_name']}"
        console.print(Panel(_format_snippet(r.get("text", "")), title=header, border_style=score_color))

    console.print(Panel(answer, title="Answer", border_style="blue"))
    console.print(f"\n[dim]Timing: {format_timing(timing)}[/dim]")

    return results


def main():
    parser = argparse.ArgumentParser(description="Query BLAS codebase in natural language")
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Natural language question (omit for interactive mode)",
    )
    parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of results")
    parser.add_argument(
        "--feature",
        choices=["explain", "docs", "translate", "patterns"],
        default=None,
        help="Feature-specific answer style",
    )
    args = parser.parse_args()

    if args.query:
        # Single query mode
        run_query_cli(args.query, k=args.top_k, feature=args.feature)
        return

    # Interactive mode
    console.print("[bold cyan]LegacyLens[/bold cyan] — Query BLAS in natural language. Type [bold]quit[/bold] to exit.\n")
    while True:
        try:
            query = console.input("[bold green]Query>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break

        results = run_query_cli(query, k=args.top_k, feature=args.feature)

        # Drill-down prompt
        if results:
            console.print("\n[dim]Enter result number (1–{}) to see full file, or press Enter to continue:[/dim]".format(len(results)))
            try:
                choice = console.input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if choice.isdigit():
                idx = int(choice)
                if 1 <= idx <= len(results):
                    file_path = results[idx - 1].get("metadata", {}).get("file_path")
                    if file_path:
                        chunks = get_full_file(file_path)
                        full_text = "\n\n".join(c.get("text", "") for c in chunks)
                        meta = chunks[0].get("metadata", {}) if chunks else {}
                        # Limit full file display to 2000 chars for very long files
                        display_text = full_text[:2000] + ("..." if len(full_text) > 2000 else "")
                        console.print(Panel(
                            Syntax(display_text, "fortran", theme="monokai", line_numbers=True),
                            title=f"Full file: {meta.get('file_name', file_path)}",
                            border_style="cyan",
                        ))


if __name__ == "__main__":
    main()
