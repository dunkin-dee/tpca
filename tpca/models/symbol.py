"""
Core data models for TPCA symbols and graphs.
"""
from dataclasses import dataclass, field
from typing import Optional
import networkx as nx


@dataclass
class Symbol:
    """
    Represents a code symbol (class, function, method, etc.).
    
    Fully qualified IDs include file path to ensure uniqueness across files.
    Example: 'src/auth.py::Auth.validate_token'
    """
    
    id: str                    # 'src/auth.py::Auth.validate_token'
    type: str                  # class|function|method|import|constant
    name: str                  # 'validate_token'
    qualified_name: str        # 'Auth.validate_token'
    file: str                  # relative path from repo root
    start_line: int
    end_line: int
    signature: str             # function/method signature
    docstring: str = ''        # first 120 chars
    parent_class: Optional[str] = None
    bases: list[str] = field(default_factory=list)  # base classes
    decorators: list[str] = field(default_factory=list)
    pagerank: float = 0.0      # filled by GraphRanker
    calls: list[str] = field(default_factory=list)  # callee names from body (duplicates = multiple sites)

    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if not isinstance(other, Symbol):
            return False
        return self.id == other.id


# Type alias for the symbol graph
# NetworkX DiGraph with Symbol IDs as nodes and edge types:
# - type='calls', weight=call_count
# - type='inherits'
# - type='imports'
# - type='member_of'
# - type='external_call', weight=0.1 (unresolved references)
SymbolGraph = nx.DiGraph


@dataclass
class PendingEdge:
    """
    Represents an unresolved edge during graph building.
    Resolved in second pass once all files are indexed.
    """
    source_id: str
    target_name: str  # unqualified name to be resolved
    edge_type: str    # 'calls', 'imports', etc.
    count: int = 1
