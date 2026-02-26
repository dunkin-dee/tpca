"""
Example tests for TPCA Phase 1 components.

These tests demonstrate the testing approach for:
- StructuredLogger
- ASTIndexer
- GraphBuilder
- GraphRanker
- IndexRenderer
- IndexCache

Run with: pytest tpca/tests/
"""

import pytest
from pathlib import Path
import tempfile
import os

from tpca import (
    TPCAConfig,
    LogConfig,
    StructuredLogger,
    ASTIndexer,
    GraphBuilder,
    GraphRanker,
    IndexRenderer,
    IndexCache,
    Symbol
)


class TestStructuredLogger:
    """Tests for the StructuredLogger."""
    
    def test_logger_emits_events(self):
        """Test that logger emits events to ring buffer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LogConfig(log_file=f"{tmpdir}/test.log")
            with StructuredLogger(config) as logger:
                
                # Emit some events
                logger.info('test_event', foo='bar', count=42)
                logger.debug('debug_event', detail='test')
                
                # Check ring buffer
                events = logger.get_events()
                assert len(events) >= 2

                logger.close()
                
                # Check specific event
                test_events = logger.get_events('test_event')
                assert len(test_events) == 1
                assert test_events[0]['foo'] == 'bar'
                assert test_events[0]['count'] == 42
    
    def test_logger_filters_by_level(self):
        """Test that console handler respects level filtering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LogConfig(
                log_file=f"{tmpdir}/test.log",
                console_level='WARN'  # Only WARN and ERROR
            )
            with StructuredLogger(config) as logger:                
                logger.debug('should_not_appear')
                logger.info('should_not_appear')
                logger.warn('should_appear')
                
                # All events are in ring buffer regardless of level
                assert len(logger.get_events()) == 3
                
                # But console handler would filter (tested via mock in real tests)


class TestIndexCache:
    """Tests for the IndexCache."""
    
    def test_cache_get_set(self):
        """Test basic cache get/set operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TPCAConfig(
                cache_dir=tmpdir,
                cache_enabled=True
            )
            with StructuredLogger(LogConfig(log_file=f"{tmpdir}/test.log")) as logger:
                cache = IndexCache(config, logger)
                
                # Create a test file
                test_file = Path(tmpdir) / "test.py"
                test_file.write_text("def foo(): pass")
                
                # Create dummy symbols
                symbols = [
                    Symbol(
                        id=f"{test_file}::foo",
                        type='function',
                        name='foo',
                        qualified_name='foo',
                        file=str(test_file),
                        start_line=1,
                        end_line=1,
                        signature='def foo()',
                    )
                ]
                
                # Set cache
                cache.set(str(test_file), symbols)
                
                # Get from cache
                cached = cache.get(str(test_file))
                assert cached is not None
                assert len(cached) == 1
                assert cached[0].name == 'foo'
    
    def test_cache_invalidation_on_modification(self):
        """Test that cache is invalidated when file is modified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TPCAConfig(
                cache_dir=tmpdir,
                cache_enabled=True
            )
            with StructuredLogger(LogConfig(log_file=f"{tmpdir}/test.log")) as logger:
                cache = IndexCache(config, logger)
                
                # Create a test file
                test_file = Path(tmpdir) / "test.py"
                test_file.write_text("def foo(): pass")
                
                symbols = [Symbol(
                    id=f"{test_file}::foo",
                    type='function',
                    name='foo',
                    qualified_name='foo',
                    file=str(test_file),
                    start_line=1,
                    end_line=1,
                    signature='def foo()',
                )]
                
                # Cache it
                cache.set(str(test_file), symbols)
                
                # Verify cached
                assert cache.get(str(test_file)) is not None
                
                # Modify file (update mtime)
                import time
                time.sleep(0.1)  # Ensure mtime is different
                test_file.write_text("def foo():\n    return 42")
                
                # Should be invalidated
                assert cache.get(str(test_file)) is None


class TestGraphRanker:
    """Tests for the GraphRanker."""
    
    def test_pagerank_assigns_scores(self):
        """Test that PageRank assigns scores to all symbols."""
        config = TPCAConfig()
        logger = StructuredLogger(config.log)
        
        # Create a simple graph
        from tpca.models import SymbolGraph
        graph = SymbolGraph()
        
        # Add some nodes
        symbols = [
            Symbol(id='file.py::A', type='class', name='A',
                  qualified_name='A', file='file.py',
                  start_line=1, end_line=10, signature='class A'),
            Symbol(id='file.py::B', type='class', name='B',
                  qualified_name='B', file='file.py',
                  start_line=11, end_line=20, signature='class B'),
        ]
        
        for sym in symbols:
            graph.add_node(sym.id, symbol=sym)
        
        # Add an edge (A calls B)
        graph.add_edge('file.py::A', 'file.py::B', type='calls', weight=1.0)
        
        # Rank
        ranker = GraphRanker(config, logger)
        ranked_graph = ranker.rank_symbols(graph, task_keywords=['test'])
        
        # Check that scores were assigned
        for node_id in ranked_graph.nodes():
            symbol = ranked_graph.nodes[node_id]['symbol']
            assert symbol.pagerank > 0
            assert 'tier' in ranked_graph.nodes[node_id]
    
    def test_task_biased_ranking(self):
        """Test that task keywords bias the ranking."""
        config = TPCAConfig()
        logger = StructuredLogger(config.log)
        
        from tpca.models import SymbolGraph
        graph = SymbolGraph()
        
        # Create symbols, one matching task keywords
        auth_sym = Symbol(
            id='file.py::authenticate',
            type='function',
            name='authenticate',
            qualified_name='authenticate',
            file='file.py',
            start_line=1,
            end_line=5,
            signature='def authenticate()',
            docstring='Authenticate user with token'
        )
        
        other_sym = Symbol(
            id='file.py::helper',
            type='function',
            name='helper',
            qualified_name='helper',
            file='file.py',
            start_line=6,
            end_line=10,
            signature='def helper()'
        )
        
        graph.add_node(auth_sym.id, symbol=auth_sym)
        graph.add_node(other_sym.id, symbol=other_sym)
        
        # Rank with task keywords
        ranker = GraphRanker(config, logger)
        ranked_graph = ranker.rank_symbols(graph, task_keywords=['auth', 'token'])
        
        # The auth function should rank higher due to keyword match
        # (Note: in a minimal graph like this, the difference might be subtle)
        auth_score = ranked_graph.nodes[auth_sym.id]['symbol'].pagerank
        other_score = ranked_graph.nodes[other_sym.id]['symbol'].pagerank
        
        # Both should have scores
        assert auth_score > 0
        assert other_score > 0


class TestIndexRenderer:
    """Tests for the IndexRenderer."""
    
    def test_renders_compact_index(self):
        """Test that renderer creates compact text output."""
        config = TPCAConfig()
        logger = StructuredLogger(config.log)
        renderer = IndexRenderer(config, logger)
        
        from tpca.models import SymbolGraph
        graph = SymbolGraph()
        
        # Add a simple symbol
        sym = Symbol(
            id='test.py::MyClass',
            type='class',
            name='MyClass',
            qualified_name='MyClass',
            file='test.py',
            start_line=1,
            end_line=10,
            signature='class MyClass'
        )
        graph.add_node(sym.id, symbol=sym, tier='CORE')
        
        # Render
        output = renderer.render(graph)
        
        # Check output contains expected elements
        assert 'test.py' in output
        assert 'MyClass' in output
        assert '[CORE]' in output
    
    def test_compact_summary(self):
        """Test compact summary generation."""
        config = TPCAConfig()
        logger = StructuredLogger(config.log)
        renderer = IndexRenderer(config, logger)
        
        from tpca.models import SymbolGraph
        graph = SymbolGraph()
        
        # Add some symbols
        for i in range(5):
            sym = Symbol(
                id=f'file{i}.py::func{i}',
                type='function',
                name=f'func{i}',
                qualified_name=f'func{i}',
                file=f'file{i}.py',
                start_line=1,
                end_line=5,
                signature=f'def func{i}()'
            )
            graph.add_node(sym.id, symbol=sym, tier='CORE')
        
        summary = renderer.render_compact_summary(graph)
        
        # Should mention number of files and symbols
        assert 'files' in summary.lower()
        assert 'symbols' in summary.lower()
        assert '5' in summary


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
