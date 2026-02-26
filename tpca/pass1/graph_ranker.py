"""
Graph Ranker - applies task-biased PageRank to symbol graphs.
"""
import networkx as nx
from typing import Optional

from ..config import TPCAConfig
from ..logging import StructuredLogger
from ..models import SymbolGraph


class GraphRanker:
    """
    Applies task-biased PageRank to rank symbols by importance.
    
    Symbols with high PageRank are architecturally central (called by many others).
    The personalization vector biases ranking toward symbols lexically related
    to the task description.
    """
    
    # Rank tier thresholds (percentiles)
    TIER_THRESHOLDS = {
        'CORE': 0.90,       # Top 10%
        'SUPPORT': 0.70,    # 70th-90th percentile
        'PERIPHERAL': 0.0   # Below 70th percentile
    }
    
    def __init__(self, config: TPCAConfig, logger: StructuredLogger):
        self.config = config
        self.logger = logger
    
    def rank_symbols(self, graph: SymbolGraph,
                    task_keywords: Optional[list[str]] = None) -> SymbolGraph:
        """
        Apply PageRank to graph and assign rank tiers to symbols.
        
        Args:
            graph: Symbol graph (will be modified in place)
            task_keywords: Keywords from task description for personalization
        
        Returns:
            The modified graph (same object)
        """
        if graph.number_of_nodes() == 0:
            self.logger.warn('graph_empty', reason='no_symbols_to_rank')
            return graph
        
        # Build personalization vector
        personalization = self._build_personalization(graph, task_keywords or [])
        
        # Compute PageRank
        try:
            pagerank_scores = nx.pagerank(
                graph,
                alpha=self.config.pagerank_alpha,
                personalization=personalization,
                max_iter=100,
                weight='weight'
            )
            
            self.logger.info('pagerank_computed',
                           nodes=len(pagerank_scores),
                           max_score=max(pagerank_scores.values()),
                           min_score=min(pagerank_scores.values()))
        
        except Exception as e:
            self.logger.error('pagerank_failed', error=str(e))
            # Fall back to uniform scores
            pagerank_scores = {node: 1.0 / graph.number_of_nodes()
                             for node in graph.nodes()}
        
        # Assign scores to symbol objects
        for node_id, score in pagerank_scores.items():
            if graph.has_node(node_id):
                symbol = graph.nodes[node_id].get('symbol')
                if symbol:
                    symbol.pagerank = score
        
        # Assign rank tiers
        self._assign_tiers(graph, pagerank_scores)
        
        return graph
    
    def _build_personalization(self, graph: SymbolGraph,
                               task_keywords: list[str]) -> dict[str, float]:
        """
        Build personalization vector for PageRank.
        
        Scores nodes based on lexical similarity to task keywords.
        """
        personalization = {}
        
        for node_id in graph.nodes():
            symbol = graph.nodes[node_id].get('symbol')
            if not symbol:
                personalization[node_id] = 0.01
                continue
            
            # Score based on keyword matches in symbol name and docstring
            text = (symbol.name + ' ' + symbol.qualified_name + ' ' +
                   symbol.docstring).lower()
            
            score = sum(1 for kw in task_keywords if kw.lower() in text)
            personalization[node_id] = max(score, 0.01)
        
        # Normalize
        total = sum(personalization.values())
        if total > 0:
            personalization = {k: v / total for k, v in personalization.items()}
        
        self.logger.debug('personalization_built',
                         keywords=task_keywords,
                         non_zero_nodes=sum(1 for v in personalization.values() if v > 0.01))
        
        return personalization
    
    def _assign_tiers(self, graph: SymbolGraph, pagerank_scores: dict[str, float]):
        """
        Assign rank tier labels (CORE, SUPPORT, PERIPHERAL) to symbols.
        
        Tiers are based on percentiles of the PageRank scores.
        """
        if not pagerank_scores:
            return
        
        # Sort scores to find percentile thresholds
        sorted_scores = sorted(pagerank_scores.values(), reverse=True)
        n = len(sorted_scores)
        
        core_threshold = sorted_scores[int(n * (1 - self.TIER_THRESHOLDS['CORE']))]
        support_threshold = sorted_scores[int(n * (1 - self.TIER_THRESHOLDS['SUPPORT']))]
        
        # Assign tiers
        tier_counts = {'CORE': 0, 'SUPPORT': 0, 'PERIPHERAL': 0}
        
        for node_id, score in pagerank_scores.items():
            if score >= core_threshold:
                tier = 'CORE'
            elif score >= support_threshold:
                tier = 'SUPPORT'
            else:
                tier = 'PERIPHERAL'
            
            graph.nodes[node_id]['tier'] = tier
            tier_counts[tier] += 1
        
        self.logger.info('tiers_assigned',
                        core=tier_counts['CORE'],
                        support=tier_counts['SUPPORT'],
                        peripheral=tier_counts['PERIPHERAL'])
    
    def get_top_symbols(self, graph: SymbolGraph, n: int = None) -> list[tuple[str, float]]:
        """
        Get top N symbols by PageRank score.
        
        Args:
            graph: Symbol graph
            n: Number of symbols to return (default: config.top_n_symbols)
        
        Returns:
            List of (symbol_id, score) tuples, sorted descending
        """
        if n is None:
            n = self.config.top_n_symbols
        
        scores = []
        for node_id in graph.nodes():
            symbol = graph.nodes[node_id].get('symbol')
            if symbol:
                scores.append((node_id, symbol.pagerank))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]
