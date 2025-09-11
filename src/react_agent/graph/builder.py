"""Graph builder for the ReAct agent."""

from __future__ import annotations

from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.context import Context
from react_agent.state import State, InputState
from react_agent.tools import TOOLS
from react_agent.graph.nodes.plan import plan
from react_agent.graph.nodes.act import act
from react_agent.graph.nodes.assess import assess
from react_agent.graph.nodes.repair import repair
from react_agent.graph.nodes.finish import finish
from react_agent.graph.nodes.progress import advance_step
from react_agent.graph.routing import (
    should_continue,
    route_after_plan,
    route_after_act,
    route_after_assess,
    route_after_repair,
)


def create_graph() -> StateGraph:
    """Construct the enhanced ReAct agent graph."""
    builder = StateGraph(State, input_schema=InputState, context_schema=Context)
    
    # Add nodes
    builder.add_node("plan", plan)
    builder.add_node("act", act)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_node("assess", assess)
    builder.add_node("repair", repair)
    builder.add_node("advance_step", advance_step)
    builder.add_node("finish", finish)
    
    # Add edges
    builder.add_conditional_edges("__start__", should_continue)
    builder.add_conditional_edges("plan", route_after_plan)
    builder.add_conditional_edges("act", route_after_act)
    builder.add_edge("tools", "assess")
    builder.add_conditional_edges("assess", route_after_assess)
    builder.add_edge("advance_step", "act")
    builder.add_conditional_edges("repair", route_after_repair)
    builder.add_edge("finish", "__end__")
    
    return builder.compile(name="Enhanced ReAct Agent")


# Export the compiled graph
graph = create_graph()