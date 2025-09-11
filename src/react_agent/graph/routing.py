"""Routing functions for the ReAct agent graph."""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage

from react_agent.state import State


def should_continue(state: State) -> Literal["plan", "act"]:
    """Route to plan if no plan exists, otherwise act."""
    if state.plan is None:
        return "plan"
    return "act"


def route_after_plan(state: State) -> Literal["act"]:
    """After planning, always proceed to action."""
    return "act"


def route_after_act(state: State) -> Literal["tools", "assess"]:
    """Route after action - to tools if tool calls exist, otherwise assess."""
    if state.messages and isinstance(state.messages[-1], AIMessage):
        if state.messages[-1].tool_calls:
            return "tools"
    return "assess"


def route_after_assess(state: State) -> Literal["advance_step", "act", "repair", "finish"]:
    """Route based on assessment outcome."""
    if not state.current_assessment:
        return "finish"
    
    if state.current_assessment.outcome == "success":
        next_index = state.step_index + 1
        if state.plan and next_index < len(state.plan.steps):
            return "advance_step"
        return "finish"
    
    elif state.current_assessment.outcome == "retry":
        if state.retry_count >= state.max_retries_per_step:
            return "repair"
        return "act"
    
    else:  # blocked
        return "repair"


def route_after_repair(state: State) -> Literal["act", "finish"]:
    """After repair, continue with action or finish if too many revisions."""
    if state.plan_revision_count >= 2:
        return "finish"
    return "act"