"""
FileWatcher — background file system watcher for live AST cache invalidation.

Uses the `watchdog` library when available; degrades gracefully if not installed.

Usage:
    watcher = FileWatcher(exclude_patterns=config.exclude_patterns)
    watcher.start("/path/to/project", callback=lambda p: cache.invalidate(p))
    # ... work ...
    watcher.stop()
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Set

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    _WATCHDOG_AVAILABLE = True
except ImportError:
    _WATCHDOG_AVAILABLE = False
    # Stubs so the class body below still parses
    Observer = None  # type: ignore
    FileSystemEventHandler = object  # type: ignore


# Extensions that trigger a cache invalidation when changed
_SOURCE_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".pyi", ".js", ".jsx", ".mjs", ".cjs",
    ".ts", ".tsx", ".mts", ".cts",
})


class _ChangeHandler(FileSystemEventHandler):  # type: ignore[misc]
    """
    Watchdog event handler that filters events to source file changes
    and calls the registered callback with the changed path.
    """

    def __init__(
        self,
        extensions: frozenset[str],
        exclude_dirs: Set[str],
        callback: Callable[[str], None],
    ) -> None:
        super().__init__()
        self._extensions = extensions
        self._exclude_dirs = exclude_dirs
        self._callback = callback

    def _should_handle(self, path: str) -> bool:
        p = Path(path)
        # Skip excluded directories anywhere in the path
        for part in p.parts:
            if part in self._exclude_dirs:
                return False
        return p.suffix.lower() in self._extensions

    def on_modified(self, event: "FileSystemEvent") -> None:
        if not event.is_directory and self._should_handle(event.src_path):
            self._callback(str(event.src_path))

    def on_created(self, event: "FileSystemEvent") -> None:
        if not event.is_directory and self._should_handle(event.src_path):
            self._callback(str(event.src_path))

    def on_moved(self, event: "FileSystemEvent") -> None:
        # Treat a rename/move as a creation of the destination
        dest = getattr(event, "dest_path", None)
        if dest and not event.is_directory and self._should_handle(dest):
            self._callback(str(dest))


class FileWatcher:
    """
    Thin wrapper around a watchdog Observer.

    Attributes:
        available:          True if watchdog is installed.
        running:            True if the watcher background thread is active.
        last_change_at:     datetime of the most recent handled event, or None.
        last_changed_path:  Path string of the most recently changed file, or None.
        source_dir:         The directory currently being watched, or None.
    """

    def __init__(
        self,
        extensions: Optional[frozenset[str]] = None,
        exclude_patterns: Optional[list[str]] = None,
    ) -> None:
        self._extensions: frozenset[str] = extensions or _SOURCE_EXTENSIONS
        self._exclude_dirs: Set[str] = set(exclude_patterns or [
            "__pycache__", ".git", "node_modules", "dist", ".venv", "venv",
            ".prism_cache",
        ])
        self._observer: Optional[Observer] = None
        self._callback: Optional[Callable[[str], None]] = None

        self.available: bool = _WATCHDOG_AVAILABLE
        self.last_change_at: Optional[datetime] = None
        self.last_changed_path: Optional[str] = None
        self.source_dir: Optional[str] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self, source_dir: str, callback: Callable[[str], None]) -> bool:
        """
        Start watching *source_dir* recursively.

        Args:
            source_dir: Absolute path to the directory to watch.
            callback:   Called with the changed file path on each source change.

        Returns:
            True if the watcher started, False if watchdog is not installed.
        """
        if not _WATCHDOG_AVAILABLE:
            return False

        # Stop any existing watch before starting a new one
        if self._observer is not None:
            self.stop()

        self._callback = callback
        self.source_dir = str(Path(source_dir).resolve())

        handler = _ChangeHandler(self._extensions, self._exclude_dirs, self._on_change)
        self._observer = Observer()
        self._observer.schedule(handler, self.source_dir, recursive=True)
        self._observer.daemon = True   # don't block process exit
        self._observer.start()
        return True

    def stop(self) -> None:
        """Stop the watcher and join the background thread."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
            self.source_dir = None

    @property
    def running(self) -> bool:
        """True if the watcher thread is alive."""
        return self._observer is not None and self._observer.is_alive()

    def status_line(self) -> str:
        """Short human-readable status for REPL display."""
        if not self.available:
            return "watchdog not installed  (pip install watchdog)"
        if not self.running:
            return "stopped"
        age = ""
        if self.last_change_at:
            delta = (datetime.now() - self.last_change_at).total_seconds()
            if delta < 60:
                age = f"  last change {int(delta)}s ago: {Path(self.last_changed_path).name}"
            else:
                age = f"  last change {int(delta // 60)}m ago"
        return f"watching {self.source_dir}{age}"

    # ── Internal ───────────────────────────────────────────────────────────────

    def _on_change(self, path: str) -> None:
        self.last_change_at = datetime.now()
        self.last_changed_path = path
        if self._callback:
            self._callback(path)
