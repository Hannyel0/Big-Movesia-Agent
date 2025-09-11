"""Enhanced ReAct agent with narration guard for forced tool calls."""

# Re-export the graph for backwards compatibility
from react_agent.graph.builder import graph, create_graph

__all__ = ["graph", "create_graph"]