"""
OutputWriter: routes synthesis output to the correct files
based on TPCAConfig.output_mode.

Supported modes:
  single_file  — all output written to one file (config.output_dir/output.md)
  mirror       — output file tree mirrors the input source tree
  per_symbol   — one file per top-level symbol
  inline       — output returned in-memory (no files written)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from ..models.output import OutputManifest, ManifestEntry


class OutputWriter:
    """
    Writes synthesis output chunks to the appropriate files,
    tracking what has been written in an OutputManifest.
    """

    SUPPORTED_MODES = ("single_file", "mirror", "per_symbol", "inline")

    def __init__(self, config, logger, source_root: Optional[str] = None, task: str = ""):
        self._config = config
        self._logger = logger
        self._source_root = Path(source_root) if source_root else None
        self._output_dir = Path(config.output_dir)
        self._mode = config.output_mode
        self._task = task
        self._manifest = OutputManifest(task=task, output_mode=self._mode)

        # In-memory accumulator (used for 'inline' mode or testing)
        self._inline_buffer: dict[str, list[str]] = {}
        # Track written files: symbol_id -> output_file path
        self._symbol_to_file: dict[str, Path] = {}

        if self._mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unknown output_mode {self._mode!r}. "
                f"Choose from {self.SUPPORTED_MODES}."
            )

        if self._mode != "inline":
            self._output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, symbol_id: str, raw_output: str) -> Optional[Path]:
        """
        Write a synthesis chunk to the appropriate output target.

        Args:
            symbol_id:  Fully-qualified symbol ID (e.g. 'src/auth.py::Auth').
            raw_output: The synthesised text for this symbol.

        Returns:
            Path to the output file, or None in 'inline' mode.
        """
        output_path = self._resolve_output_path(symbol_id)
        self._symbol_to_file[symbol_id] = output_path

        if self._mode == "inline":
            self._inline_buffer.setdefault(str(output_path), []).append(raw_output)
            self._logger.debug(
                "output_inline", symbol_id=symbol_id, chars=len(raw_output)
            )
            return None

        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Append mode — multiple chunks may go to the same file
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(raw_output)
            if not raw_output.endswith("\n"):
                f.write("\n")

        self._logger.info(
            "output_written",
            symbol_id=symbol_id,
            path=str(output_path),
            chars=len(raw_output),
        )
        return output_path

    def flush_inline(self) -> dict[str, str]:
        """Return the complete in-memory output for 'inline' mode."""
        return {k: "\n".join(v) for k, v in self._inline_buffer.items()}

    def build_manifest(
        self,
        task: str,
        chunks: list[dict],
    ) -> OutputManifest:
        """
        Build the final OutputManifest from recorded chunks.

        Args:
            task:   The original task description.
            chunks: List of chunk dicts from the OutputLog.
        """
        manifest = OutputManifest(task=task, output_mode=self._mode)

        # Group chunks by output file
        file_chunks: dict[str, list[dict]] = {}
        for chunk in chunks:
            sym_id = chunk["symbol_id"]
            out_path = str(self._symbol_to_file.get(sym_id, "inline"))
            file_chunks.setdefault(out_path, []).append(chunk)

        for out_path, ch_list in file_chunks.items():
            # Find source file from symbol ID of first chunk
            first_sym = ch_list[0]["symbol_id"]
            source_file = self._source_from_symbol(first_sym)
            total_tokens = sum(c.get("token_count", 0) for c in ch_list)

            entry = ManifestEntry(
                source_file=source_file,
                output_file=out_path,
                symbols_processed=[c["symbol_id"] for c in ch_list],
                chunk_count=len(ch_list),
                token_count=total_tokens,
                status="complete",
            )
            manifest.add_file(entry)

        manifest.mark_complete()
        self._manifest = manifest
        return manifest

    # ── Path resolution ───────────────────────────────────────────────────────

    def _resolve_output_path(self, symbol_id: str) -> Path:
        """Map a symbol_id to an output file path based on output_mode."""
        source_file = self._source_from_symbol(symbol_id)
        stem = Path(source_file).stem  # e.g. 'auth'

        if self._mode == "single_file":
            return self._output_dir / "output.md"

        elif self._mode == "mirror":
            # Replicate directory structure under output_dir
            try:
                if self._source_root:
                    rel = Path(source_file).relative_to(self._source_root)
                else:
                    rel = Path(source_file)
            except ValueError:
                rel = Path(source_file)
            return self._output_dir / rel.with_suffix(".md")

        elif self._mode == "per_symbol":
            # One file per symbol (sanitised name)
            safe_name = symbol_id.replace("/", "_").replace("::", "__").replace(" ", "_")
            return self._output_dir / f"{safe_name}.md"

        else:  # inline
            return Path("inline") / f"{stem}.md"

    @staticmethod
    def _source_from_symbol(symbol_id: str) -> str:
        """Extract the source file path from a fully-qualified symbol ID."""
        # symbol_id format: 'path/to/file.py::ClassName.method'
        if "::" in symbol_id:
            return symbol_id.split("::")[0]
        return symbol_id

    # ── Phase 3 additions ─────────────────────────────────────────────────────

    def get_output(self) -> dict[str, str]:
        """Return the complete in-memory output (inline mode) or last-chunk buffer."""
        return self.flush_inline()

    def finalize(self) -> None:
        """Mark the manifest complete and save it to disk."""
        self._manifest.mark_complete()
        self._save_manifest()

    def save_partial(self) -> None:
        """Persist the manifest in its current partial state (for resume on interrupt)."""
        self._save_manifest()

    def mark_all_complete(self) -> None:
        """Mark every manifest entry as complete."""
        for entry in self._manifest.files:
            entry.status = "complete"

    def mark_file_complete(self, source_file: str) -> None:
        """Mark one source file's manifest entry as complete."""
        entry = self._manifest.get_entry(source_file)
        if entry:
            entry.status = "complete"
            self._manifest.upsert_entry(entry)

    def _save_manifest(self) -> None:
        """Write manifest.json to the cache directory."""
        import os as _os
        cache_dir = getattr(self._config, "cache_dir", ".tpca_cache")
        manifest_path = Path(cache_dir) / "manifest.json"
        try:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            self._manifest.save(str(manifest_path))
            self._logger.debug("manifest_saved", path=str(manifest_path))
        except OSError as exc:
            self._logger.warn("manifest_save_failed", error=str(exc))

    @classmethod
    def load_manifest(cls, path: str) -> OutputManifest:
        """Load a manifest from disk for the resume path in TPCAOrchestrator."""
        return OutputManifest.load(path)
    
    @property
    def manifest(self) -> OutputManifest:
        """Public accessor for the current manifest."""
        return self._manifest