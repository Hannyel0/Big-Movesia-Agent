"""Progress node for the ReAct agent."""

from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.state import (
    ExecutionPlan,
    PlanStep,
    State,
    StepStatus,
)


async def advance_step(
    state: State,
    runtime: Runtime[Context]
) -> Dict[str, Any]:
    """Advance to the next step with progress update."""
    if not state.plan:
        return {}
    
    # Add current step to completed steps
    completed = state.completed_steps.copy()
    if state.step_index not in completed:
        completed.append(state.step_index)
    
    # Move to next step
    next_index = state.step_index + 1
    
    # Update the current step status
    updated_steps = []
    for i, step in enumerate(state.plan.steps):
        if i == state.step_index:
            updated_step = PlanStep(
                description=step.description,
                tool_name=step.tool_name,
                success_criteria=step.success_criteria,
                dependencies=step.dependencies,
                status=StepStatus.SUCCEEDED,
                attempts=step.attempts,
                error_messages=step.error_messages
            )
            updated_steps.append(updated_step)
        else:
            updated_steps.append(step)
    
    updated_plan = ExecutionPlan(
        goal=state.plan.goal,
        steps=updated_steps,
        metadata=state.plan.metadata
    )
    
    # Optional: Add progress message for multi-step plans
    progress_message = None
    if len(state.plan.steps) > 2:  # Only for multi-step plans
        progress_message = AIMessage(content=f"✅ Step {state.step_index + 1} completed. Moving to step {next_index + 1}...")
    
    result = {
        "plan": updated_plan,
        "step_index": next_index,
        "completed_steps": completed,
        "retry_count": 0,  # Reset retry count for new step
        "current_assessment": None
    }
    
    if progress_message:
        result["messages"] = [progress_message]
    
    return result


async def increment_retry(
    state: State,
    runtime: Runtime[Context]
) -> Dict[str, Any]:
    """Increment retry count for the current step."""
    return {
        "retry_count": state.retry_count + 1,
        "current_assessment": None  # Clear previous assessment
    }