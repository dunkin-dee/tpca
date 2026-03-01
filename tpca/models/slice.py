"""
Slice and SliceRequest data models for Pass 2 context packaging.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Slice:
    """A fetched source code slice for a single symbol."""
    symbol_id: str
    source: str              # Full source lines (or signature-only if truncated)
    token_count: int
    truncated: bool = False  # True if signature-only fallback was used
    start_line: int = 0
    end_line: int = 0

    def __repr__(self) -> str:
        trunc = " [TRUNCATED]" if self.truncated else ""
        return f"Slice({self.symbol_id!r}, tokens={self.token_count}{trunc})"


@dataclass
class SliceRequest:
    """
    The structured output from ContextPlanner: which symbols to fetch and why.
    """
    primary_symbols: list[str] = field(default_factory=list)
    supporting_symbols: list[str] = field(default_factory=list)
    rationale: str = ""

    @property
    def all_symbols(self) -> list[str]:
        return self.primary_symbols + self.supporting_symbols

    def to_dict(self) -> dict:
        return {
            "primary_symbols": self.primary_symbols,
            "supporting_symbols": self.supporting_symbols,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SliceRequest":
        return cls(
            primary_symbols=data.get("primary_symbols", []),
            supporting_symbols=data.get("supporting_symbols", []),
            rationale=data.get("rationale", ""),
        )
