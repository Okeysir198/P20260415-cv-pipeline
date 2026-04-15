"""LangGraph graph definition for generative augmentation pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from langgraph.graph import StateGraph, END

from utils.langgraph_common import should_continue
from core.p03_generative_aug.nodes import (
    GenAugmentState,
    scan_node,
    segment_batch,
    inpaint_batch,
    validate_batch,
    write_batch,
    aggregate_node,
)


def build_graph():
    """Build and compile the generative augmentation LangGraph.

    Graph structure:
        START -> scan -> segment_batch -> inpaint_batch -> validate_batch -> write_batch
                                                                                 |
                                                      has_more? --continue--> segment_batch
                                                                --aggregate-> aggregate -> END

    Returns:
        Compiled LangGraph ready for invocation via ``.invoke()``.
    """
    graph = StateGraph(GenAugmentState)

    # Add nodes
    graph.add_node("scan", scan_node)
    graph.add_node("segment_batch", segment_batch)
    graph.add_node("inpaint_batch", inpaint_batch)
    graph.add_node("validate_batch", validate_batch)
    graph.add_node("write_batch", write_batch)
    graph.add_node("aggregate", aggregate_node)

    # Add edges
    graph.set_entry_point("scan")
    graph.add_edge("scan", "segment_batch")
    graph.add_edge("segment_batch", "inpaint_batch")
    graph.add_edge("inpaint_batch", "validate_batch")
    graph.add_edge("validate_batch", "write_batch")

    # Conditional edge: continue or aggregate
    graph.add_conditional_edges(
        "write_batch",
        should_continue,
        {
            "continue": "segment_batch",
            "aggregate": "aggregate",
        },
    )

    graph.add_edge("aggregate", END)

    return graph.compile()
