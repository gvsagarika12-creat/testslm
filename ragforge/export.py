"""Dump the corpus as JSONL — the raw-text shape continued pretraining consumes."""
from __future__ import annotations

import json
from pathlib import Path

from ragforge.store import VectorStore


def export_jsonl(store: VectorStore, out_path: Path) -> int:
    """Write one JSON object per chunk. Returns the number of records written."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in store.iter_chunks():
            handle.write(
                json.dumps(
                    {
                        "text": chunk.text,
                        "source": chunk.source_filename,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
    return written
