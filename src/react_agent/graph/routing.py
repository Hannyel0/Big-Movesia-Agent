"""Enhanced routing functions for the ReAct agent graph with complexity-based routing."""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage

from react_agent.state import State


def should_continue(state: State) -> Literal["classify", "act"]:
    """Route to classify if no plan exists, otherwise act."""
    if state.plan is None:
        return "classify"
    return "act"


def route_after_classify(state: State) -> Literal["direct_act", "simple_plan", "plan"]:
    """Route based on complexity classification."""
    complexity_level = state.runtime_metadata.get("complexity_level", "complex_plan")
    
    if complexity_level == "direct":
        return "direct_act"
    elif complexity_level == "simple_plan":
        return "simple_plan"
    else:
        return "plan"


def route_after_plan(state: State) -> Literal["act"]:
    """After planning, always proceed to action."""
    return "act"


def route_after_simple_plan(state: State) -> Literal["act"]:
    """After simple planning, proceed to action."""
    return "act"


def route_after_direct_act(state: State) -> Literal["tools", "__end__"]:
    """Route after direct action - to tools if tool calls exist, otherwise end."""
    if state.messages and isinstance(state.messages[-1], AIMessage):
        if state.messages[-1].tool_calls:
            return "tools"
    return "__end__"


def route_after_act(state: State) -> Literal["tools", "assess"]:
    """Route after action - to tools if tool calls exist, otherwise assess."""
    if state.messages and isinstance(state.messages[-1], AIMessage):
        if state.messages[-1].tool_calls:
            return "tools"
    return "assess"


def route_after_assess(state: State) -> Literal["advance_step", "increment_retry", "repair", "finish"]:
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
        return "increment_retry"
    
    else:  # blocked
        return "repair"


def route_after_repair(state: State) -> Literal["act", "finish"]:
    """After repair, continue with action or finish if too many revisions."""
    if state.plan_revision_count >= 2:
        return "finish"
    return "act"


def route_classification_aware(state: State) -> Literal["direct_act", "simple_plan", "plan", "act"]:
    """Unified routing function that considers both classification and plan state."""
    # If we have a plan, proceed with normal execution
    if state.plan is not None:
        return "act"
    
    # No plan, check classification
    complexity_level = state.runtime_metadata.get("complexity_level")
    
    if complexity_level == "direct":
        return "direct_act"
    elif complexity_level == "simple_plan":
        return "simple_plan"
    elif complexity_level == "complex_plan":
        return "plan"
    else:
        # No classification yet, shouldn't happen but fallback to classification
        return "plan"  # Conservative fallback