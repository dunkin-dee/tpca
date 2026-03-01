"""
Output data models: OutputLog, OutputChunk, OutputManifest, ManifestEntry.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import json
from datetime import datetime, timezone


@dataclass
class OutputChunk:
    """A single completed synthesis chunk recorded in the OutputLog."""
    chunk_id: int
    symbol_id: str
    summary: str        # One-line LLM-generated summary
    status: str         # complete | partial | failed
    token_count: int    # Tokens in the raw output

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "symbol_id": self.symbol_id,
            "summary": self.summary,
            "status": self.status,
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OutputChunk":
        return cls(**data)


@dataclass
class OutputLog:
    """
    Compact working memory for the synthesis loop.
    Passed to each subsequent synthesis call instead of the full prior output.
    """
    entries: list[OutputChunk] = field(default_factory=list)

    def add(self, chunk: OutputChunk) -> None:
        self.entries.append(chunk)

    def completed_symbols(self) -> set[str]:
        return {e.symbol_id for e in self.entries if e.status == "complete"}

    def render_compact(self) -> str:
        """Render the OutputLog as a compact text block (~50-100 tokens per entry)."""
        if not self.entries:
            return ""
        lines = ["## COMPLETED WORK (do not repeat — reference only)"]
        for entry in self.entries:
            lines.append(f"[{entry.chunk_id}] {entry.symbol_id}  →  {entry.summary}")
        return "\n".join(lines)

    def total_tokens(self) -> int:
        return sum(e.token_count for e in self.entries)

    def to_dict(self) -> dict:
        return {"entries": [e.to_dict() for e in self.entries]}

    @classmethod
    def from_dict(cls, data: dict) -> "OutputLog":
        return cls(entries=[OutputChunk.from_dict(e) for e in data.get("entries", [])])


@dataclass
class ManifestEntry:
    """Per-file record in the OutputManifest."""
    source_file: str
    output_file: str
    symbols_processed: list[str]
    chunk_count: int
    token_count: int
    status: str         # complete | partial | skipped

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "output_file": self.output_file,
            "symbols_processed": self.symbols_processed,
            "chunk_count": self.chunk_count,
            "token_count": self.token_count,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ManifestEntry":
        return cls(**data)


@dataclass
class OutputManifest:
    """Tracks all output files for a multi-file synthesis task."""
    task: str
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: Optional[str] = None
    output_mode: str = "single_file"
    files: list[ManifestEntry] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def mark_complete(self) -> None:
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def add_file(self, entry: ManifestEntry) -> None:
        self.files.append(entry)

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "output_mode": self.output_mode,
            "files": [f.to_dict() for f in self.files],
            "stats": self.stats,
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "OutputManifest":
        obj = cls(task=data["task"])
        obj.started_at = data.get("started_at", obj.started_at)
        obj.completed_at = data.get("completed_at")
        obj.output_mode = data.get("output_mode", "single_file")
        obj.files = [ManifestEntry.from_dict(f) for f in data.get("files", [])]
        obj.stats = data.get("stats", {})
        return obj

    @classmethod
    def load(cls, path: str) -> "OutputManifest":
        with open(path) as f:
            return cls.from_dict(json.load(f))
