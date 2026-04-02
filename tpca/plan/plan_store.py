"""
PlanStore — atomic JSON persistence for SessionPlan.

Writes to .tpca_plan.json in the project root.
Uses write-to-temp + os.replace() for atomicity.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from .plan_model import SessionPlan


class PlanStore:
    FILENAME = ".tpca_plan.json"

    def __init__(self, project_root: str) -> None:
        self._root = Path(project_root).resolve()
        self._path = self._root / self.FILENAME

    def load(self) -> Optional[SessionPlan]:
        """Load the persisted plan. Returns None if none exists or file is corrupt."""
        if not self._path.is_file():
            return None
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return SessionPlan.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

    def save(self, plan: SessionPlan) -> None:
        """Atomically write plan to disk (tmp file → os.replace)."""
        from datetime import datetime
        plan.updated_at = datetime.utcnow().isoformat()
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(plan.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(str(tmp), str(self._path))

    def clear(self) -> None:
        """Delete the persisted plan file if it exists."""
        if self._path.is_file():
            self._path.unlink()

    def exists(self) -> bool:
        return self._path.is_file()

    @property
    def path(self) -> Path:
        return self._path
