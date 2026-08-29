"""Command line entrypoint. Thin wrapper over ragforge.pipeline."""
from __future__ import annotations

from pathlib import Path

import typer

from ragforge.export import export_jsonl
from ragforge.pipeline import build_pipeline

app = typer.Typer(help="Local PDF ingestion and retrieval.", no_args_is_help=True)


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="A PDF file or a directory of PDFs."),
    chunk_size: int = typer.Option(None, help="Tokens per chunk."),
    overlap: int = typer.Option(None, help="Tokens of overlap between chunks."),
    recursive: bool = typer.Option(True, help="Descend into subdirectories."),
    force: bool = typer.Option(False, help="Re-ingest even if already stored."),
) -> None:
    """Ingest a PDF or a folder of PDFs."""
    pipeline = build_pipeline()
    report = pipeline.ingest_path(
        path, recursive=recursive, chunk_size=chunk_size, overlap=overlap, force=force
    )

    if not report.results:
        typer.echo("No PDF files found.")
        raise typer.Exit(code=1)

    for result in report.results:
        typer.echo(f"[{result.status:>8}] {result.path.name} — {result.message}")

    typer.echo(
        f"\n{len(report.ingested)} ingested, {len(report.skipped)} skipped, "
        f"{len(report.failed)} failed, {report.total_chunks} chunks written."
    )
    if report.failed and not report.ingested:
        raise typer.Exit(code=1)


@app.command()
def search(
    query: str = typer.Argument(..., help="What to look for."),
    k: int = typer.Option(5, help="Number of results."),
) -> None:
    """Search the corpus and print ranked chunks."""
    hits = build_pipeline().search(query, k=k)
    if not hits:
        typer.echo("No results.")
        return
    for rank, hit in enumerate(hits, start=1):
        pages = (
            f"p{hit.page_start}"
            if hit.page_start == hit.page_end
            else f"p{hit.page_start}-{hit.page_end}"
        )
        typer.echo(f"\n{rank}. {hit.score:.3f}  {hit.source_filename}  {pages}")
        typer.echo(f"   {hit.text[:300]}")


@app.command()
def stats() -> None:
    """Show what is currently in the store."""
    for key, value in build_pipeline().stats().items():
        typer.echo(f"{key}: {value}")


@app.command()
def migrate() -> None:
    """Copy an existing ChromaDB corpus into PostgreSQL.

    Vectors are carried over as-is, so nothing is re-embedded.
    """
    from ragforge.migrate import migrate_chroma_to_postgres

    moved = migrate_chroma_to_postgres()
    typer.echo(f"Migrated {moved} chunks from ChromaDB into PostgreSQL.")


@app.command()
def export(
    out_path: Path = typer.Argument(..., help="Destination .jsonl file."),
) -> None:
    """Export every chunk as JSONL for later training use."""
    written = export_jsonl(build_pipeline().store, out_path)
    typer.echo(f"Wrote {written} records to {out_path}")


if __name__ == "__main__":
    app()
