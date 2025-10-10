"""Enhanced routing functions with micro-retry support and completion detection."""

from __future__ import annotations
from typing import Literal
import json
from langchain_core.messages import AIMessage, ToolMessage
from react_agent.state import State


def route_after_assess(state: State) -> Literal["advance_step", "increment_retry", "micro_retry", "error_recovery", "finish"]:
    """Enhanced routing with micro-retry support and absolute completion priority."""
    # Safety checks
    if not state.current_assessment:
        return "finish"
    
    if not state.plan or not state.plan.steps:
        return "finish"
    
    # NEW: Check for micro-retry flag first (highest priority for transient errors)
    if getattr(state, "should_micro_retry", False):
        return "micro_retry"
    
    # ABSOLUTE PRIORITY: Check if we're on or past the last step
    last_step_index = len(state.plan.steps) - 1
    is_on_or_past_last_step = state.step_index >= last_step_index
    
    # If we're on the last step AND it succeeded, ALWAYS finish
    if is_on_or_past_last_step and state.current_assessment.outcome == "success":
        return "finish"
    
    # If we somehow went past the last step, force finish
    if state.step_index >= len(state.plan.steps):
        return "finish"
    
    # Check for error recovery need (only if not completing)
    if getattr(state, "needs_error_recovery", False) or state.runtime_metadata.get("needs_error_recovery"):
        return "error_recovery"
    
    # Handle success on non-final steps
    if state.current_assessment.outcome == "success":
        return "advance_step"
    
    # Handle retry logic
    if state.current_assessment.outcome == "retry":
        if state.retry_count >= state.max_retries_per_step:
            return "error_recovery"
        return "increment_retry"
    
    # Handle blocked steps
    if state.current_assessment.outcome == "blocked":
        return "error_recovery"
    
    # Fallback to finish
    return "finish"


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


def route_after_error_recovery(state: State) -> Literal["act", "finish"]:
    """Route after error recovery execution."""
    if state.plan and state.step_index < len(state.plan.steps):
        return "act"
    return "finish"


def route_after_micro_retry(state: State) -> Literal["act"]:
    """Route after micro-retry - always back to act to retry the tool."""
    return "act"


def route_after_tools(state: State) -> Literal["check_file_approval", "assess"]:
    """
    Route after tool execution to check if approval is needed.
    
    This function checks the most recent tool message to see if it contains
    a file operation that requires human approval.
    
    Returns:
        "check_file_approval" if the tool result needs approval
        "assess" for normal tool execution flow
    """
    # Find the last tool message
    last_tool_message = None
    for msg in reversed(state.messages):
        if isinstance(msg, ToolMessage):
            last_tool_message = msg
            break
    
    if not last_tool_message:
        # No tool message found, proceed to assessment
        return "assess"
    
    # Parse tool result to check for approval needs
    try:
        result = json.loads(last_tool_message.content)
        
        # If the tool returned needs_approval=True, route to approval handler
        if result.get("needs_approval"):
            print(f"🔍 [Router] Tool result needs approval, routing to check_file_approval")
            print(f"   Tool: {last_tool_message.name}")
            print(f"   Operation: {result.get('approval_data', {}).get('operation')}")
            print(f"   File: {result.get('approval_data', {}).get('file_path')}")
            return "check_file_approval"
    except (json.JSONDecodeError, AttributeError) as e:
        # Not JSON or no content, proceed to assessment
        print(f"⚠️  [Router] Could not parse tool result as JSON: {e}")
        pass
    
    # Normal flow - go to assessment
    return "assess"


def route_classification_aware(state: State) -> Literal["direct_act", "simple_plan", "plan", "act"]:
    """Unified routing function that considers both classification and plan state."""
    if state.plan is not None:
        return "act"
    
    complexity_level = state.runtime_metadata.get("complexity_level")
    
    if complexity_level == "direct":
        return "direct_act"
    elif complexity_level == "simple_plan":
        return "simple_plan"
    elif complexity_level == "complex_plan":
        return "plan"
    else:
        return "plan"