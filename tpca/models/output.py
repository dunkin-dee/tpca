"""
Output data models: OutputLog, OutputChunk, OutputManifest, ManifestEntry.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import json
from datetime import datetime, timezone
from pathlib import Path


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

    @property
    def chunks(self) -> list[OutputChunk]:
        """Alias for entries — used by Phase 3 fallback code and tests."""
        return self.entries

    def completed_symbols(self) -> set[str]:
        return {e.symbol_id for e in self.entries if e.status == "complete"}

    def render_compact(self) -> str:
        """Render the OutputLog as a compact text block (~50-100 tokens per entry)."""
        if not self.entries:
            return ""
        lines = ["## COMPLETED WORK (do not repeat — reference only)"]
        for entry in self.entries:
            lines.append(
                f"[{entry.chunk_id}] {entry.symbol_id}  →  {entry.summary}")
        return "\n".join(lines)

    def total_tokens(self) -> int:
        return sum(e.token_count for e in self.entries)

    def to_dict(self) -> dict:
        return {"entries": [e.to_dict() for e in self.entries]}

    @classmethod
    def from_dict(cls, data: dict) -> "OutputLog":
        return cls(entries=[OutputChunk.from_dict(e) for e in data.get("entries", [])])

    @classmethod
    def from_manifest(cls, manifest: "OutputManifest") -> "OutputLog":
        """
        Reconstruct an OutputLog from a saved OutputManifest.

        Used by TPCAOrchestrator when resuming an interrupted run to restore
        cross-file consistency context without re-processing complete files.
        """
        log = cls()
        chunk_id = 0
        for entry in manifest.files:
            if entry.status == "complete":
                for sym_id in entry.symbols_processed:
                    log.add(OutputChunk(
                        chunk_id=chunk_id,
                        symbol_id=sym_id,
                        summary=f"Previously completed — see {entry.output_file}.",
                        status="complete",
                        token_count=entry.token_count // max(
                            len(entry.symbols_processed), 1),
                    ))
                    chunk_id += 1
        return log


@dataclass
class ManifestEntry:
    """Per-file record in the OutputManifest."""
    source_file: str
    output_file: str
    symbols_processed: list[str] = field(default_factory=list)
    chunk_count: int = 0
    token_count: int = 0
    status: str = "partial"   # complete | partial | skipped

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
        return cls(
            source_file=data["source_file"],
            output_file=data["output_file"],
            symbols_processed=data.get("symbols_processed", []),
            chunk_count=data.get("chunk_count", 0),
            token_count=data.get("token_count", 0),
            status=data.get("status", "partial"),
        )


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

    # ── Phase 3 additions ─────────────────────────────────────────────────────

    def get_entry(self, source_file: str) -> "Optional[ManifestEntry]":
        """Return the ManifestEntry for source_file, or None if not found."""
        for entry in self.files:
            if entry.source_file == source_file:
                return entry
        return None

    def upsert_entry(self, entry: ManifestEntry) -> None:
        """Update an existing entry with the same source_file, or append."""
        for i, e in enumerate(self.files):
            if e.source_file == entry.source_file:
                self.files[i] = entry
                return
        self.files.append(entry)

    def incomplete_files(self) -> list[ManifestEntry]:
        """Return entries that are not yet fully complete."""
        return [e for e in self.files if e.status != "complete"]

    def is_done(self) -> bool:
        """True only when completed_at is set AND every entry is complete."""
        return bool(self.completed_at) and all(
            e.status == "complete" for e in self.files
        )

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
        Path(path).parent.mkdir(parents=True, exist_ok=True)
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
