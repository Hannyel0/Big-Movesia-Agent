"""Enhanced ReAct agent graph with planning and assessment capabilities."""

from __future__ import annotations

import json
from typing import Any, Dict, Literal

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.prompts import (
    ACT_PROMPT,
    ASSESSMENT_PROMPT,
    FINAL_SUMMARY_PROMPT,
    PLANNING_PROMPT,
    REPAIR_PROMPT,
)
from react_agent.state import (
    AssessmentOutcome,
    ExecutionPlan,
    InputState,
    PlanStep,
    State,
    StepStatus,
)
from react_agent.tools import TOOLS
from react_agent.utils import get_message_text, get_model

# Core node functions

async def plan(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Create an execution plan for the user's request."""
    context = runtime.context
    
    # Use planning model or fall back to main model
    model = get_model(context.planning_model or context.model)
    
    # Extract the user query from messages
    user_message = None
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            user_message = get_message_text(msg)
            break
    
    if not user_message:
        return {"messages": [AIMessage(content="No user query found to plan for.")]}
    
    # Prepare tools information for planning
    tools_info = "\n".join([
        f"- {tool.name}: {tool.description}" 
        for tool in TOOLS
    ])
    
    # Create planning prompt
    planning_prompt = PLANNING_PROMPT.format(tools_info=tools_info)
    
    # Generate plan
    messages = [
        {"role": "system", "content": planning_prompt},
        {"role": "user", "content": f"Create a plan to: {user_message}"}
    ]
    
    response = await model.ainvoke(messages)
    plan_content = get_message_text(response)
    
    # Parse the plan (simplified - in reality you'd want structured parsing)
    # For now, create a basic plan structure
    plan = ExecutionPlan(
        goal=user_message,
        steps=[
            PlanStep(
                description=f"Execute user request: {user_message}",
                success_criteria="User request is fulfilled with accurate information",
                tool_name="tavily_search_results_json" if "search" in user_message.lower() else None
            )
        ]
    )
    
    return {
        "plan": plan,
        "step_index": 0,
        "retry_count": 0,
        "messages": [AIMessage(content=f"Plan created:\n{plan_content}")]
    }


async def act(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Execute the current step in the plan."""
    context = runtime.context
    model = get_model(context.model)
    
    if not state.plan or state.step_index >= len(state.plan.steps):
        return {"messages": [AIMessage(content="No valid step to execute.")]}
    
    current_step = state.plan.steps[state.step_index]
    
    # Prepare execution context
    execution_context = {
        "current_step": current_step.description,
        "success_criteria": current_step.success_criteria,
        "step_number": state.step_index + 1,
        "total_steps": len(state.plan.steps),
        "available_tools": [tool.name for tool in TOOLS]
    }
    
    # Create action prompt
    action_prompt = ACT_PROMPT.format(
        execution_context=json.dumps(execution_context, indent=2)
    )
    
    # Get conversation history for context
    conversation_messages = []
    for msg in state.messages:
        if isinstance(msg, HumanMessage):
            conversation_messages.append({"role": "user", "content": get_message_text(msg)})
        elif isinstance(msg, AIMessage):
            conversation_messages.append({"role": "assistant", "content": get_message_text(msg)})
    
    # Prepare messages for the model
    messages = [
        {"role": "system", "content": action_prompt},
        *conversation_messages,
        {"role": "user", "content": f"Execute this step: {current_step.description}"}
    ]
    
    # Bind tools to model and invoke
    model_with_tools = model.bind_tools(TOOLS)
    response = await model_with_tools.ainvoke(messages)
    
    # Update step status
    updated_steps = list(state.plan.steps)
    updated_steps[state.step_index] = PlanStep(
        description=current_step.description,
        tool_name=current_step.tool_name,
        success_criteria=current_step.success_criteria,
        dependencies=current_step.dependencies,
        status=StepStatus.IN_PROGRESS,
        attempts=current_step.attempts + 1,
        error_messages=current_step.error_messages
    )
    
    updated_plan = ExecutionPlan(
        goal=state.plan.goal,
        steps=updated_steps,
        metadata=state.plan.metadata
    )
    
    return {
        "plan": updated_plan,
        "messages": [response],
        "total_tool_calls": state.total_tool_calls + (1 if response.tool_calls else 0)
    }


async def assess(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Assess whether the current step succeeded."""
    context = runtime.context
    model = get_model(context.assessment_model or context.model)
    
    if not state.plan or state.step_index >= len(state.plan.steps):
        return {}
    
    current_step = state.plan.steps[state.step_index]
    
    # Get the last tool result or AI response
    last_message = state.messages[-1] if state.messages else None
    last_result = ""
    
    if isinstance(last_message, AIMessage):
        last_result = get_message_text(last_message)
    elif isinstance(last_message, ToolMessage):
        last_result = get_message_text(last_message)
    
    # Create assessment prompt
    assessment_prompt = f"""{ASSESSMENT_PROMPT}

Step to assess: {current_step.description}
Success criteria: {current_step.success_criteria}
Actual result: {last_result}

Based on the success criteria and actual result, provide your assessment as JSON:
{{
    "outcome": "success|retry|blocked",
    "reason": "explanation of your assessment",
    "fix": "suggested fix if retry is needed (optional)",
    "confidence": 0.8
}}
"""
    
    messages = [
        {"role": "system", "content": assessment_prompt}
    ]
    
    response = await model.ainvoke(messages)
    assessment_text = get_message_text(response)
    
    # Parse assessment (simplified - would want better error handling)
    try:
        # Try to extract JSON from the response
        import re
        json_match = re.search(r'\{.*\}', assessment_text, re.DOTALL)
        if json_match:
            assessment_data = json.loads(json_match.group())
            assessment = AssessmentOutcome(
                outcome=assessment_data.get("outcome", "retry"),
                reason=assessment_data.get("reason", "Assessment completed"),
                fix=assessment_data.get("fix"),
                confidence=assessment_data.get("confidence", 0.8)
            )
        else:
            # Fallback assessment
            if "success" in assessment_text.lower():
                outcome = "success"
            elif "blocked" in assessment_text.lower():
                outcome = "blocked"
            else:
                outcome = "retry"
            
            assessment = AssessmentOutcome(
                outcome=outcome,
                reason=assessment_text,
                confidence=0.6
            )
    except Exception:
        # Default to retry if parsing fails
        assessment = AssessmentOutcome(
            outcome="retry",
            reason="Failed to parse assessment response",
            confidence=0.5
        )
    
    return {
        "current_assessment": assessment,
        "total_assessments": state.total_assessments + 1
    }


async def repair(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Replan after encountering issues."""
    context = runtime.context
    model = get_model(context.planning_model or context.model)
    
    if not state.plan:
        return {}
    
    # Gather information about what went wrong
    current_step = state.plan.steps[state.step_index]
    assessment = state.current_assessment
    
    repair_context = {
        "original_goal": state.plan.goal,
        "failed_step": current_step.description,
        "failure_reason": assessment.reason if assessment else "Unknown failure",
        "completed_steps": [
            state.plan.steps[i].description 
            for i in state.completed_steps
        ],
        "remaining_steps": [
            step.description 
            for i, step in enumerate(state.plan.steps) 
            if i > state.step_index
        ]
    }
    
    # Create repair prompt
    repair_prompt = f"""{REPAIR_PROMPT}

Context: {json.dumps(repair_context, indent=2)}

Create a revised plan that works around the failure and achieves the original goal.
"""
    
    messages = [
        {"role": "system", "content": repair_prompt},
        {"role": "user", "content": f"Fix the plan for: {state.plan.goal}"}
    ]
    
    response = await model.ainvoke(messages)
    repair_content = get_message_text(response)
    
    # Create revised plan (simplified)
    revised_steps = []
    
    # Keep completed steps
    for i in state.completed_steps:
        if i < len(state.plan.steps):
            step = state.plan.steps[i]
            revised_steps.append(PlanStep(
                description=step.description,
                tool_name=step.tool_name,
                success_criteria=step.success_criteria,
                dependencies=step.dependencies,
                status=StepStatus.SUCCEEDED,
                attempts=step.attempts,
                error_messages=step.error_messages
            ))
    
    # Add a new step to replace the failed one
    revised_steps.append(PlanStep(
        description=f"Alternative approach: {current_step.description}",
        success_criteria=current_step.success_criteria,
        tool_name=None  # Let the act node decide
    ))
    
    revised_plan = ExecutionPlan(
        goal=state.plan.goal,
        steps=revised_steps,
        metadata={"revision_count": state.plan_revision_count + 1}
    )
    
    return {
        "plan": revised_plan,
        "step_index": len(state.completed_steps),  # Start from first unfinished step
        "retry_count": 0,
        "current_assessment": None,
        "plan_revision_count": state.plan_revision_count + 1,
        "messages": [AIMessage(content=f"Plan revised:\n{repair_content}")]
    }


async def finish(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Provide final summary and completion."""
    context = runtime.context
    model = get_model(context.model)
    
    # Prepare summary context
    summary_context = {
        "goal": state.plan.goal if state.plan else "No plan available",
        "completed_steps": len(state.completed_steps),
        "total_steps": len(state.plan.steps) if state.plan else 0,
        "plan_revisions": state.plan_revision_count,
        "tool_calls": state.total_tool_calls,
        "assessments": state.total_assessments
    }
    
    messages = [
        {"role": "system", "content": FINAL_SUMMARY_PROMPT},
        {"role": "user", "content": f"Summarize the completion of: {summary_context['goal']}"}
    ]
    
    response = await model.ainvoke(messages)
    
    return {
        "messages": [response]
    }


# Routing functions

def should_continue(state: State) -> Literal["plan", "act"]:
    """Route to plan if no plan exists, otherwise act."""
    if state.plan is None:
        return "plan"
    return "act"


def route_after_plan(state: State) -> Literal["act"]:  # pylint: disable=unused-argument
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
        return "act"
    
    if state.current_assessment.outcome == "success":
        # Check if there are more steps after advancing
        next_index = state.step_index + 1
        if state.plan and next_index < len(state.plan.steps):
            return "advance_step"  # Advance to next step
        return "finish"  # No more steps, finish
    
    elif state.current_assessment.outcome == "retry":
        if state.retry_count >= state.max_retries_per_step:
            return "repair"
        return "act"
    
    else:  # blocked
        return "repair"


def route_after_repair(state: State) -> Literal["act", "finish"]:
    """After repair, continue with action or finish if too many revisions."""
    if state.plan_revision_count >= 3:  # Max revisions exceeded
        return "finish"
    return "act"


# Step advancement function

async def advance_step(
    state: State,
    runtime: Runtime[Context]  # pylint: disable=unused-argument
) -> Dict[str, Any]:
    """Advance to the next step after successful completion."""
    if not state.plan:
        return {}
    
    # Add current step to completed steps
    completed = state.completed_steps.copy()
    if state.step_index not in completed:
        completed.append(state.step_index)
    
    # Move to next step
    next_index = state.step_index + 1
    
    # Update the current step status (create new steps list to avoid mutation)
    updated_steps = []
    for i, step in enumerate(state.plan.steps):
        if i == state.step_index:
            # Create updated step
            updated_step = PlanStep(
                description=step.description,
                tool_name=step.tool_name,
                success_criteria=step.success_criteria,
                dependencies=step.dependencies,
                status=StepStatus.SUCCEEDED,  # Mark as succeeded
                attempts=step.attempts,
                error_messages=step.error_messages
            )
            updated_steps.append(updated_step)
        else:
            updated_steps.append(step)
    
    # Create updated plan
    updated_plan = ExecutionPlan(
        goal=state.plan.goal,
        steps=updated_steps,
        metadata=state.plan.metadata
    )
    
    return {
        "plan": updated_plan,
        "step_index": next_index,
        "completed_steps": completed,
        "retry_count": 0,
        "current_assessment": None
    }


# Graph construction

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