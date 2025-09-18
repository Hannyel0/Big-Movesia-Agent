"""Simplified progress node with absolute bounds checking."""

from __future__ import annotations
from typing import Any, Dict
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime
from react_agent.context import Context
from react_agent.state import ExecutionPlan, PlanStep, State, StepStatus


async def advance_step(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Advance to next step with absolute bounds checking and completion detection."""
    
    # Basic validation
    if not state.plan or not state.plan.steps:
        return {"messages": [AIMessage(content="ERROR: No plan to advance.")]}
    
    # Bounds checking
    if state.step_index < 0:
        return {"messages": [AIMessage(content="ERROR: Invalid negative step index.")]}
    
    if state.step_index >= len(state.plan.steps):
        return {"messages": [AIMessage(content="ERROR: Step index beyond plan bounds.")]}
    
    # Add current step to completed list
    completed = state.completed_steps.copy()
    if state.step_index not in completed:
        completed.append(state.step_index)
    
    # Calculate next step index
    next_index = state.step_index + 1
    
    # CRITICAL: If next step would be beyond bounds, we're done
    if next_index >= len(state.plan.steps):
        return {
            "completed_steps": completed,
            "current_assessment": None,
            "messages": [AIMessage(content="All steps completed successfully!")]
        }
    
    # Mark current step as succeeded
    updated_steps = list(state.plan.steps)
    current_step = updated_steps[state.step_index]
    
    updated_steps[state.step_index] = PlanStep(
        description=current_step.description,
        tool_name=current_step.tool_name,
        success_criteria=current_step.success_criteria,
        dependencies=current_step.dependencies,
        status=StepStatus.SUCCEEDED,
        attempts=current_step.attempts,
        error_messages=current_step.error_messages
    )
    
    updated_plan = ExecutionPlan(
        goal=state.plan.goal,
        steps=updated_steps,
        metadata=state.plan.metadata
    )
    
    # Create progress message
    progress_message = None
    if len(state.plan.steps) > 2:
        completed_step_num = state.step_index + 1  # 1-based for display
        next_step_num = next_index + 1  # 1-based for display
        next_step_description = state.plan.steps[next_index].description
        
        progress_message = AIMessage(
            content=f"Step {completed_step_num} completed. Moving to step {next_step_num}: {next_step_description}"
        )
    
    result = {
        "plan": updated_plan,
        "step_index": next_index,
        "completed_steps": completed,
        "retry_count": 0,
        "current_assessment": None
    }
    
    if progress_message:
        result["messages"] = [progress_message]
    
    return result


async def increment_retry(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Increment retry count for the current step."""
    return {
        "retry_count": state.retry_count + 1,
        "current_assessment": None
    }