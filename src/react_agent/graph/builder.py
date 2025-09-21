"""Enhanced graph builder with tiered error recovery and micro-retry support."""

from __future__ import annotations

from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.context import Context
from react_agent.state import State, InputState
from react_agent.tools import TOOLS
from react_agent.graph.nodes.classify import classify
from react_agent.graph.nodes.direct_act import direct_act
from react_agent.graph.nodes.simple_plan import simple_plan
from react_agent.graph.nodes.plan import plan
from react_agent.graph.nodes.act import act
from react_agent.graph.nodes.assess import assess
from react_agent.graph.nodes.finish import finish
from react_agent.graph.nodes.progress import advance_step, increment_retry
from react_agent.graph.nodes.error_recovery import execute_error_recovery
from react_agent.graph.nodes.micro_retry import micro_retry  # NEW IMPORT
from react_agent.graph.routing import (
    should_continue,
    route_after_classify,
    route_after_plan,
    route_after_simple_plan,
    route_after_direct_act,
    route_after_act,
    route_after_assess,
    route_after_error_recovery,
    route_after_micro_retry,  # NEW IMPORT
)


def create_graph() -> StateGraph:
    """Construct the ReAct agent graph with tiered error recovery and micro-retry support."""
    builder = StateGraph(State, input_schema=InputState, context_schema=Context)
    
    # Add all nodes
    builder.add_node("classify", classify)
    builder.add_node("direct_act", direct_act)
    builder.add_node("simple_plan", simple_plan)
    builder.add_node("plan", plan)
    builder.add_node("act", act)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_node("assess", assess)
    builder.add_node("error_recovery", execute_error_recovery)
    builder.add_node("micro_retry", micro_retry)  # NEW NODE
    builder.add_node("advance_step", advance_step)
    builder.add_node("increment_retry", increment_retry)
    builder.add_node("finish", finish)
    
    # Add edges
    builder.add_edge("__start__", "classify")
    builder.add_conditional_edges("classify", route_after_classify)
    builder.add_conditional_edges("direct_act", route_after_direct_act)
    builder.add_conditional_edges("simple_plan", route_after_simple_plan)
    builder.add_conditional_edges("plan", route_after_plan)
    builder.add_conditional_edges("act", route_after_act)
    builder.add_edge("tools", "assess")
    builder.add_conditional_edges("assess", route_after_assess)
    builder.add_conditional_edges("error_recovery", route_after_error_recovery)
    builder.add_conditional_edges("micro_retry", route_after_micro_retry)  # NEW EDGE
    builder.add_edge("advance_step", "act")
    builder.add_edge("increment_retry", "act")
    builder.add_edge("finish", "__end__")
    
    return builder.compile(name="ReAct Agent with Tiered Error Recovery")


# Export the compiled graph
graph = create_graph()