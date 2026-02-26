"""
TPCA Phase 1 Demo
Demonstrates the complete Pass 1 pipeline: indexing, ranking, and rendering.

Requirements:
    pip install tree-sitter-python networkx

Usage:
    python demo_phase1.py
"""
import sys
from pathlib import Path

# Add tpca to path
sys.path.insert(0, str(Path(__file__).parent))

from tpca import (
    TPCAConfig,
    LogConfig,
    StructuredLogger,
    ASTIndexer,
    GraphBuilder,
    GraphRanker,
    IndexRenderer,
    IndexCache
)


def main():
    """Run the Phase 1 demo."""
    print("=" * 60)
    print("TPCA Phase 1 Demo - AST Indexing & Graph Ranking")
    print("=" * 60)
    print()
    
    # Configure TPCA
    config = TPCAConfig(
        languages=['python'],
        cache_enabled=True,
        cache_dir='.tpca_cache',
        pagerank_alpha=0.85,
        log=LogConfig(
            console_level='INFO',
            log_file='.tpca_cache/demo.log'
        )
    )
    
    # Initialize logger
    logger = StructuredLogger(config.log)
    logger.info('demo_start', phase='1')
    
    # Initialize cache
    cache = IndexCache(config, logger)
    print(f"📦 Cache initialized: {cache.cache_dir}")
    print()
    
    # Initialize Pass 1 components
    indexer = ASTIndexer(config, logger, cache)
    graph_builder = GraphBuilder(config, logger)
    ranker = GraphRanker(config, logger)
    renderer = IndexRenderer(config, logger)
    
    # Target directory to index
    sample_dir = Path(__file__).parent / 'tests' / 'fixtures' / 'sample_codebase'
    
    print(f"📂 Indexing directory: {sample_dir}")
    print()
    
    # Step 1: Index source files
    print("Step 1: Parsing Python files with AST Indexer...")
    symbols = indexer.index(str(sample_dir))
    print(f"  ✓ Found {len(symbols)} symbols")
    print()
    
    # Step 2: Build symbol graph
    print("Step 2: Building symbol relationship graph...")
    graph = graph_builder.build(symbols)
    print(f"  ✓ Graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print()
    
    # Step 3: Rank symbols with PageRank
    print("Step 3: Ranking symbols with task-biased PageRank...")
    task_keywords = ['auth', 'validate', 'token', 'user']
    graph = ranker.rank_symbols(graph, task_keywords)
    print(f"  ✓ Symbols ranked (task keywords: {task_keywords})")
    print()
    
    # Show top symbols
    print("📊 Top 10 symbols by PageRank:")
    top_symbols = ranker.get_top_symbols(graph, n=10)
    for i, (symbol_id, score) in enumerate(top_symbols, 1):
        symbol = graph.nodes[symbol_id].get('symbol')
        tier = graph.nodes[symbol_id].get('tier', '')
        print(f"  {i:2}. [{tier:11}] {score:.6f}  {symbol_id}")
    print()
    
    # Step 4: Render compact index
    print("Step 4: Rendering compact index...")
    compact_index = renderer.render(graph)
    print(f"  ✓ Index rendered ({len(compact_index)} chars, ~{len(compact_index)//4} tokens)")
    print()
    
    # Display the compact index
    print("=" * 60)
    print("COMPACT INDEX OUTPUT")
    print("=" * 60)
    print(compact_index)
    print()
    
    # Summary
    summary = renderer.render_compact_summary(graph)
    print("=" * 60)
    print(f"Summary: {summary}")
    print("=" * 60)
    print()
    
    # Cache statistics
    cache_stats = cache.get_stats()
    print("Cache Statistics:")
    print(f"  • Cached files: {cache_stats['cached_files']}")
    print(f"  • Total cached symbols: {cache_stats['total_symbols']}")
    print(f"  • Cache directory: {cache_stats['cache_dir']}")
    print()
    
    # Show log events
    print("Recent Log Events:")
    events = logger.get_events()[-10:]  # Last 10 events
    for event in events:
        level = event.get('level', 'INFO')
        event_name = event.get('event', '')
        print(f"  [{level:5}] {event_name}")
    print()
    
    logger.info('demo_complete', phase='1',
                symbols_indexed=len(symbols),
                graph_size=graph.number_of_nodes())
    
    print("✅ Demo complete!")
    print(f"   Check {config.log.log_file} for detailed logs")


if __name__ == '__main__':
    try:
        main()
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nPlease install required packages:")
        print("  pip install tree-sitter-python networkx")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
