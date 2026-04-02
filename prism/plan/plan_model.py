"""
Plan data model for PRISM coding-assistant sessions.

Hierarchy:
    SessionPlan
      └── PlanSection (recursive: sub_sections)
            ├── PlanEvaluation  (set after evaluator runs)
            └── WorkerSummary   (set after worker agent runs)
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ── PlanEvaluation ────────────────────────────────────────────────────────────

@dataclass
class PlanEvaluation:
    score: float             # 0.0–1.0 composite
    completeness: float      # all required work covered?
    granularity: float       # fits in one context window? (1.0 = yes)
    consistency: float       # coherent with plan neighbours?
    code_correctness: float  # 13B+ only: diff reviewed (0.0 for plan eval)
    test_coverage: float     # 13B+ only: new code has tests? (0.0 for plan eval)
    critique: str            # ≤200 chars
    recommendation: str      # "APPROVE" | "REVISE" | "SPLIT"

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "completeness": self.completeness,
            "granularity": self.granularity,
            "consistency": self.consistency,
            "code_correctness": self.code_correctness,
            "test_coverage": self.test_coverage,
            "critique": self.critique,
            "recommendation": self.recommendation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlanEvaluation":
        return cls(
            score=float(d.get("score", 0.5)),
            completeness=float(d.get("completeness", 0.5)),
            granularity=float(d.get("granularity", 1.0)),
            consistency=float(d.get("consistency", 0.5)),
            code_correctness=float(d.get("code_correctness", 0.0)),
            test_coverage=float(d.get("test_coverage", 0.0)),
            critique=str(d.get("critique", ""))[:200],
            recommendation=str(d.get("recommendation", "APPROVE")),
        )


# ── WorkerSummary ─────────────────────────────────────────────────────────────

@dataclass
class WorkerSummary:
    section_id: str
    status: str                       # "COMPLETE" | "PARTIAL" | "FAILED"
    brief: str                        # ≤150 chars, past tense
    detail: str                       # ≤500 chars, decisions made and why
    files_changed: list[str]
    symbols_touched: list[str]
    interfaces_changed: list[str] = field(default_factory=list)
    new_symbols: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    token_cost: int = 0
    test_result: Optional[str] = None  # "PASS" | "FAIL: <reason>" | None

    def to_dict(self) -> dict:
        return {
            "section_id": self.section_id,
            "status": self.status,
            "brief": self.brief,
            "detail": self.detail,
            "files_changed": self.files_changed,
            "symbols_touched": self.symbols_touched,
            "interfaces_changed": self.interfaces_changed,
            "new_symbols": self.new_symbols,
            "assumptions": self.assumptions,
            "blockers": self.blockers,
            "token_cost": self.token_cost,
            "test_result": self.test_result,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WorkerSummary":
        return cls(
            section_id=d["section_id"],
            status=d["status"],
            brief=d["brief"],
            detail=d.get("detail", ""),
            files_changed=list(d.get("files_changed", [])),
            symbols_touched=list(d.get("symbols_touched", [])),
            interfaces_changed=list(d.get("interfaces_changed", [])),
            new_symbols=list(d.get("new_symbols", [])),
            assumptions=list(d.get("assumptions", [])),
            blockers=list(d.get("blockers", [])),
            token_cost=int(d.get("token_cost", 0)),
            test_result=d.get("test_result"),
        )


# ── PlanSection ───────────────────────────────────────────────────────────────

@dataclass
class PlanSection:
    id: str
    title: str
    description: str
    scope_symbols: list[str] = field(default_factory=list)
    scope_files: list[str] = field(default_factory=list)
    estimated_tokens: int = 0
    status: str = "PENDING"   # PENDING|IN_PROGRESS|COMPLETE|NEEDS_REVISION|BLOCKED
    assigned_agent_id: Optional[str] = None
    sub_sections: list["PlanSection"] = field(default_factory=list)
    evaluation: Optional[PlanEvaluation] = None
    worker_summary: Optional[WorkerSummary] = None
    created_at: str = ""
    updated_at: str = ""

    def is_leaf(self) -> bool:
        return not self.sub_sections

    def all_leaves(self) -> list["PlanSection"]:
        if self.is_leaf():
            return [self]
        leaves: list[PlanSection] = []
        for sub in self.sub_sections:
            leaves.extend(sub.all_leaves())
        return leaves

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "scope_symbols": self.scope_symbols,
            "scope_files": self.scope_files,
            "estimated_tokens": self.estimated_tokens,
            "status": self.status,
            "assigned_agent_id": self.assigned_agent_id,
            "sub_sections": [s.to_dict() for s in self.sub_sections],
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
            "worker_summary": self.worker_summary.to_dict() if self.worker_summary else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlanSection":
        return cls(
            id=d["id"],
            title=d["title"],
            description=d["description"],
            scope_symbols=list(d.get("scope_symbols", [])),
            scope_files=list(d.get("scope_files", [])),
            estimated_tokens=int(d.get("estimated_tokens", 0)),
            status=d.get("status", "PENDING"),
            assigned_agent_id=d.get("assigned_agent_id"),
            sub_sections=[
                PlanSection.from_dict(s) for s in d.get("sub_sections", [])
            ],
            evaluation=(
                PlanEvaluation.from_dict(d["evaluation"])
                if d.get("evaluation") else None
            ),
            worker_summary=(
                WorkerSummary.from_dict(d["worker_summary"])
                if d.get("worker_summary") else None
            ),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )


# ── SessionPlan ───────────────────────────────────────────────────────────────

@dataclass
class SessionPlan:
    plan_id: str
    task: str
    project_root: str
    index_snapshot_hash: str
    sections: list[PlanSection]
    global_style_notes: str = ""
    status: str = "PLANNING"   # PLANNING|EVALUATING|EXECUTING|COMPLETE|FAILED
    model_preset: str = ""
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def new(
        cls,
        task: str,
        project_root: str,
        index_hash: str = "",
        preset: str = "",
    ) -> "SessionPlan":
        now = datetime.utcnow().isoformat()
        return cls(
            plan_id=str(uuid.uuid4()),
            task=task,
            project_root=project_root,
            index_snapshot_hash=index_hash,
            sections=[],
            status="PLANNING",
            model_preset=preset,
            created_at=now,
            updated_at=now,
        )

    def all_leaf_sections(self) -> list[PlanSection]:
        leaves: list[PlanSection] = []
        for s in self.sections:
            leaves.extend(s.all_leaves())
        return leaves

    def pending_leaves(self) -> list[PlanSection]:
        return [s for s in self.all_leaf_sections() if s.status == "PENDING"]

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "task": self.task,
            "project_root": self.project_root,
            "index_snapshot_hash": self.index_snapshot_hash,
            "sections": [s.to_dict() for s in self.sections],
            "global_style_notes": self.global_style_notes,
            "status": self.status,
            "model_preset": self.model_preset,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SessionPlan":
        return cls(
            plan_id=d["plan_id"],
            task=d["task"],
            project_root=d["project_root"],
            index_snapshot_hash=d.get("index_snapshot_hash", ""),
            sections=[PlanSection.from_dict(s) for s in d.get("sections", [])],
            global_style_notes=d.get("global_style_notes", ""),
            status=d.get("status", "PLANNING"),
            model_preset=d.get("model_preset", ""),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )
