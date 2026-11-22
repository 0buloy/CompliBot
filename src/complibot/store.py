"""On-disk index: chunks.jsonl + meta.json (+ embeddings.npy when dense search is on).

Deliberately boring: JSON lines and a numpy matrix. The BM25 index is rebuilt in
memory at load time (it takes milliseconds for a few thousand chunks), so the
only expensive artefact we persist is the embedding matrix.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .schemas import Chunk

CHUNKS_FILE = "chunks.jsonl"
META_FILE = "meta.json"
EMB_FILE = "embeddings.npy"


@dataclass
class IndexData:
    chunks: list[Chunk]
    embeddings: np.ndarray | None = None  # shape (n_chunks, dim), L2-normalised
    meta: dict = field(default_factory=dict)


def save_index(index_dir: Path, data: IndexData) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    with (index_dir / CHUNKS_FILE).open("w", encoding="utf-8") as f:
        for c in data.chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    emb_path = index_dir / EMB_FILE
    if data.embeddings is not None:
        np.save(emb_path, data.embeddings.astype(np.float32))
    elif emb_path.exists():
        emb_path.unlink()  # never leave stale vectors next to new chunks
    meta = {
        **data.meta,
        "n_chunks": len(data.chunks),
        "has_embeddings": data.embeddings is not None,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (index_dir / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_index(index_dir: Path) -> IndexData:
    path = index_dir / CHUNKS_FILE
    if not path.exists():
        raise FileNotFoundError(f"No index at {index_dir}. Run `complibot ingest` first.")
    with path.open(encoding="utf-8") as f:
        chunks = [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]
    meta_path = index_dir / META_FILE
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    emb_path = index_dir / EMB_FILE
    embeddings = np.load(emb_path) if emb_path.exists() else None
    if embeddings is not None and len(embeddings) != len(chunks):
        raise ValueError("embeddings.npy does not match chunks.jsonl; re-run `complibot ingest`.")
    return IndexData(chunks=chunks, embeddings=embeddings, meta=meta)
