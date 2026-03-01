"""
TPCA Phase 2 Demo
Demonstrates the complete two-pass pipeline: Pass 1 indexing + Pass 2 synthesis.

Requirements:
    pip install tree-sitter-python networkx anthropic tiktoken

Set ANTHROPIC_API_KEY to run with real LLM synthesis.
Without the key, the demo runs Pass 1 only and shows what would be sent to the LLM.

Usage:
    python demo_phase2.py

    # With real LLM synthesis:
    ANTHROPIC_API_KEY=sk-ant-... python demo_phase2.py

    # Override the task:
    TPCA_TASK="Explain the authentication flow" python demo_phase2.py
"""
import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from tpca import (
    TPCAConfig,
    LogConfig,
    StructuredLogger,
    ASTIndexer,
    GraphBuilder,
    GraphRanker,
    IndexRenderer,
    IndexCache,
    LLMClient,
    ContextPlanner,
    SliceFetcher,
    OutputChunker,
    OutputWriter,
    SynthesisAgent,
    TPCAOrchestrator,
)
from tpca.models.output import OutputLog
from tpca.models.slice import SliceRequest


def banner(title: str, char: str = "=") -> None:
    width = 65
    print(char * width)
    print(f"  {title}")
    print(char * width)
    print()


def section(title: str) -> None:
    print(f"\n{'─' * 65}")
    print(f"  {title}")
    print(f"{'─' * 65}")


def main():
    # ── Configuration ──────────────────────────────────────────────────────────
    banner("TPCA Phase 2 Demo — Two-Pass Context Agent")

    task = os.environ.get(
        "TPCA_TASK",
        "Document every public method in the codebase with its parameters "
        "and return type. Describe what each method does in 1–2 sentences.",
    )
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    has_llm = bool(api_key)

    config = TPCAConfig(
        languages=["python"],
        cache_enabled=True,
        cache_dir=".tpca_cache",
        pagerank_alpha=0.85,
        model_context_window=8192,
        context_budget_pct=0.65,
        reader_model="claude-haiku-4-5-20251001",
        synthesis_model="claude-sonnet-4-6",
        max_planner_retries=3,
        output_mode="inline",  # Return output in-memory for demo
        max_synthesis_iterations=20,
        log=LogConfig(
            console_level="INFO",
            log_file=".tpca_cache/demo_phase2.log",
        ),
    )

    print(f"📋 Task: {task[:80]}{'...' if len(task) > 80 else ''}")
    print(f"🔑 LLM Available: {'✅ Yes' if has_llm else '❌ No (Pass 1 only mode)'}")
    print(f"🎯 Output mode: {config.output_mode}")
    print(f"📐 Context budget: {int(config.model_context_window * config.context_budget_pct)} tokens")
    print()

    # ── Pass 1: Index and Rank ─────────────────────────────────────────────────
    section("PASS 1 — Deterministic AST Indexing (zero LLM calls)")

    logger = StructuredLogger(config.log)
    cache = IndexCache(config, logger)
    indexer = ASTIndexer(config, logger, cache)
    graph_builder = GraphBuilder(config, logger)
    ranker = GraphRanker(config, logger)
    renderer = IndexRenderer(config, logger)

    sample_dir = Path(__file__).parent / "tests" / "fixtures" / "sample_codebase"
    print(f"📂 Source: {sample_dir}")
    print()

    print("Step 1: Parsing Python files with Tree-sitter AST...")
    symbols = indexer.index(str(sample_dir))
    print(f"  ✓ Indexed {len(symbols)} symbols")

    print("Step 2: Building cross-file symbol graph...")
    graph = graph_builder.build(symbols)
    print(f"  ✓ Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    print("Step 3: Ranking with task-biased PageRank...")
    keywords = ["auth", "validate", "token", "router", "hash", "user"]
    graph = ranker.rank_symbols(graph, keywords)
    print(f"  ✓ Ranked (keywords: {keywords})")

    print("Step 4: Rendering compact index...")
    compact_index = renderer.render(graph)
    token_estimate = len(compact_index) // 4
    print(f"  ✓ Index: {len(compact_index)} chars (~{token_estimate} tokens)")
    print()

    # Show top symbols
    section("Top 10 Ranked Symbols")
    top = ranker.get_top_symbols(graph, n=10)
    for i, (sym_id, score) in enumerate(top, 1):
        tier = graph.nodes[sym_id].get("tier", "")
        short_id = sym_id.split("/")[-1] if "/" in sym_id else sym_id
        print(f"  {i:2}. [{tier:11}] {score:.6f}  {short_id}")
    print()

    # Show the compact index
    section("Compact Index (what the LLM sees)")
    print(compact_index)

    # ── Pass 2: Context Planning & Synthesis ───────────────────────────────────
    section("PASS 2 — LLM-Driven Context Planning & Synthesis")

    llm_client = LLMClient(config, logger)

    if not has_llm:
        print("⚠️  No ANTHROPIC_API_KEY found — demonstrating Pass 2 structure only.")
        print()
        _demo_pass2_structure(config, logger, llm_client, graph, compact_index, task)
        _show_stats_pass1_only(symbols, graph, compact_index, config)
        return

    # Full Pass 2 with real LLM
    print("🚀 Running full synthesis pipeline with Anthropic API...")
    print()

    planner = ContextPlanner(config, logger, llm_client)
    fetcher = SliceFetcher(config, logger, llm_client)
    agent = SynthesisAgent(config, logger, llm_client, planner, fetcher)

    result = agent.run(
        task=task,
        compact_index=compact_index,
        graph=graph,
        source_root=str(sample_dir),
    )

    # ── Show results ───────────────────────────────────────────────────────────
    section("Synthesis Output")
    for key, content in result.output.items():
        print(f"\n{'='*65}")
        print(f"  {key}")
        print(f"{'='*65}")
        print(content)

    section("Output Log (Working Memory)")
    print(result.output_log.render_compact())

    section("Performance Stats")
    _print_stats(result.stats, symbols, graph, compact_index)

    section("Recent Log Events")
    events = logger.get_events()[-15:]
    for ev in events:
        level = ev.get("level", "INFO")
        event_name = ev.get("event", "")
        extras = {k: v for k, v in ev.items()
                  if k not in ("ts", "level", "event") and v is not None}
        extras_str = ""
        if extras:
            key_vals = [f"{k}={v}" for k, v in list(extras.items())[:3]]
            extras_str = f"  [{', '.join(key_vals)}]"
        print(f"  [{level:5}] {event_name}{extras_str}")

    print()
    print("✅ Phase 2 demo complete!")
    print(f"   Logs: {config.log.log_file}")


def _demo_pass2_structure(config, logger, llm_client, graph, compact_index, task):
    """Show what Pass 2 would do without making LLM calls."""

    budget = int(config.model_context_window * config.context_budget_pct)

    section("Context Planning Prompt Preview (would be sent to LLM)")
    from tpca.pass2.context_planner import CONTEXT_PLANNING_PROMPT
    preview = CONTEXT_PLANNING_PROMPT.format(
        task=task,
        budget_tokens=budget,
        compact_index=compact_index,
    )
    print(preview[:1200])
    if len(preview) > 1200:
        print(f"\n  ... [{len(preview) - 1200} more chars]")
    print()

    section("Example SliceRequest (what the LLM would return)")
    # Pick the top 3 symbols from the graph as an illustrative example
    from tpca.pass1.graph_ranker import GraphRanker
    ranker = GraphRanker(config, logger)
    top = ranker.get_top_symbols(graph, n=5)
    example_request = SliceRequest(
        primary_symbols=[sym_id for sym_id, _ in top[:3]],
        supporting_symbols=[sym_id for sym_id, _ in top[3:5]],
        rationale="[This would be the LLM's rationale for requesting these symbols]",
    )
    print(json.dumps(example_request.to_dict(), indent=2))
    print()

    section("OutputChunker — Processing Order Preview")
    from tpca.pass2.output_chunker import OutputChunker
    from tpca.models.output import OutputLog
    chunker = OutputChunker(
        graph=graph,
        output_log=OutputLog(),
        config=config,
        logger=logger,
        slice_request=example_request,
    )
    print(f"  Symbols to process: {chunker.total_count}")
    print(f"  Processing order (topological):")
    for i, sym_id in enumerate(chunker._topo_order[:8]):
        short = sym_id.split("/")[-1] if "/" in sym_id else sym_id
        print(f"    {i+1}. {short}")
    print()

    section("Synthesis Prompt Preview (would be sent for each chunk)")
    from tpca.pass2.synthesis_agent import SYNTHESIS_PROMPT
    first_sym = example_request.primary_symbols[0] if example_request.primary_symbols else "sym"
    prompt_preview = SYNTHESIS_PROMPT.format(
        task=task,
        compact_index=compact_index[:500] + "\n  [...]",
        slices="### auth.py::Auth\n```python\nclass Auth:\n    def validate_token(self, token): ...\n```",
        rationale=example_request.rationale,
        prior_output_log="",
        current_symbol=first_sym,
    )
    print(prompt_preview[:800])
    print("\n  [...]")
    print()

    section("OutputLog Format (bounded working memory between calls)")
    from tpca.models.output import OutputLog, OutputChunk
    example_log = OutputLog()
    example_log.add(OutputChunk(
        chunk_id=0,
        symbol_id="tests/fixtures/sample_codebase/auth.py::Auth",
        summary="Documented __init__, validate_token, refresh_token. validate_token raises ValueError on expired tokens.",
        status="complete",
        token_count=312,
    ))
    example_log.add(OutputChunk(
        chunk_id=1,
        symbol_id="tests/fixtures/sample_codebase/utils.py::hash_password",
        summary="Documented hash_password. Uses bcrypt; returns hex digest string.",
        status="complete",
        token_count=148,
    ))
    print(example_log.render_compact())
    log_tokens = llm_client.count_tokens(example_log.render_compact())
    print(f"\n  ↑ This log = ~{log_tokens} tokens (vs full output which could be thousands)")
    print()


def _print_stats(stats: dict, symbols, graph, compact_index):
    items = [
        ("Symbols indexed", stats.get("symbols_indexed", len(symbols))),
        ("Symbols requested by LLM", stats.get("symbols_requested", "?")),
        ("Slices fetched", stats.get("slices_fetched", "?")),
        ("Tokens sent to LLM", stats.get("tokens_sent_to_llm", "?")),
        ("Output chunks", stats.get("output_chunks", "?")),
        ("OutputLog tokens", stats.get("output_log_tokens", "?")),
        ("LLM calls", stats.get("llm_calls", "?")),
        ("Compression ratio", f"{stats.get('compression_ratio', '?')}x"),
        ("Total time", f"{stats.get('total_time_ms', '?')}ms"),
    ]
    for label, value in items:
        print(f"  {label:<30} {value}")


def _show_stats_pass1_only(symbols, graph, compact_index, config):
    section("Pass 1 Statistics")
    print(f"  Symbols indexed:        {len(symbols)}")
    print(f"  Graph nodes:            {graph.number_of_nodes()}")
    print(f"  Graph edges:            {graph.number_of_edges()}")
    print(f"  Index size:             {len(compact_index)} chars")
    print(f"  Approx index tokens:    ~{len(compact_index) // 4}")
    budget = int(config.model_context_window * config.context_budget_pct)
    print(f"  Context budget:         {budget} tokens")
    print()
    print("💡 Set ANTHROPIC_API_KEY to run the full synthesis pipeline.")
    print()


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nInstall required packages:")
        print("  pip install tree-sitter-python networkx anthropic tiktoken")
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        traceback.print_exc()
