"""LangGraph graph definition for auto-annotation pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from langgraph.graph import END, StateGraph

from core.p01_auto_annotate.nodes import (
    AutoAnnotateState,
    aggregate_node,
    annotate_batch,
    nms_filter_batch,
    scan_node,
    validate_batch,
    write_batch,
)
from utils.langgraph_common import should_continue


def build_graph():
    """Build and compile the auto-annotation LangGraph.

    Graph structure:
        START -> scan -> annotate_batch -> validate_batch -> nms_filter_batch -> write_batch
                                                                                     |
                                                          has_more? --continue--> annotate_batch
                                                                    --aggregate-> aggregate -> END

    Returns:
        Compiled LangGraph ready for invocation via ``.invoke()``.
    """
    graph = StateGraph(AutoAnnotateState)

    # Add nodes
    graph.add_node("scan", scan_node)
    graph.add_node("annotate_batch", annotate_batch)
    graph.add_node("validate_batch", validate_batch)
    graph.add_node("nms_filter_batch", nms_filter_batch)
    graph.add_node("write_batch", write_batch)
    graph.add_node("aggregate", aggregate_node)

    # Add edges
    graph.set_entry_point("scan")
    graph.add_edge("scan", "annotate_batch")
    graph.add_edge("annotate_batch", "validate_batch")
    graph.add_edge("validate_batch", "nms_filter_batch")
    graph.add_edge("nms_filter_batch", "write_batch")

    # Conditional edge: continue or aggregate
    graph.add_conditional_edges(
        "write_batch",
        should_continue,
        {
            "continue": "annotate_batch",
            "aggregate": "aggregate",
        },
    )

    graph.add_edge("aggregate", END)

    return graph.compile()
