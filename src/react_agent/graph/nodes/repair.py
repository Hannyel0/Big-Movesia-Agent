"""Repair node for the ReAct agent."""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.state import (
    ExecutionPlan,
    PlanStep,
    State,
    StructuredExecutionPlan,
    StructuredPlanStep,
    ToolName,
)
from react_agent.utils import get_model


def create_smart_default_plan(user_goal: str) -> List[PlanStep]:
    """Create an intelligent default plan based on the user's goal."""
    # Import here to avoid circular imports
    from react_agent.graph.nodes.plan import create_tool_aware_plan, analyze_user_goal
    
    analysis = analyze_user_goal(user_goal)
    return create_tool_aware_plan(user_goal, analysis)


async def repair(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Replan after encountering issues using structured outputs with context awareness."""
    context = runtime.context
    model = get_model(context.planning_model or context.model)
    
    if not state.plan:
        return {}
    
    # Create a user-friendly message about trying a different approach with context
    retry_info = f" (after {state.retry_count} retries)" if state.retry_count > 0 else ""
    user_message = f"Let me try a different approach{retry_info} to better help you with this."
    
    # Gather information about what went wrong
    current_step = state.plan.steps[state.step_index]
    assessment = state.current_assessment
    
    # Include more context about previous attempts
    failure_context = ""
    if assessment:
        failure_context = f"\nIssue: {assessment.reason}"
        if assessment.fix:
            failure_context += f"\nPrevious fix suggestion: {assessment.fix}"
    
    if current_step.error_messages:
        failure_context += f"\nRecent errors: {'; '.join(current_step.error_messages[-3:])}"
    
    repair_request = f"""The current execution plan needs adjustment for: "{state.plan.goal}"

Failed Step: {current_step.description}
Recommended Tool: {current_step.tool_name}
Retry Count: {state.retry_count}{failure_context}

Create a revised plan that:
1. Addresses the specific issue encountered
2. Uses different tools or approaches where appropriate
3. Maintains focus on the original goal
4. Has clear, achievable steps
5. Considers what has already been attempted

VALID TOOL NAMES: search, get_project_info, create_asset, write_file, edit_project_config, get_script_snippets, compile_and_test, scene_management

Provide a complete revised execution plan using only the valid tool names above."""
    
    # Static repair prompt with tool constraints - cacheable
    static_repair_content = """You are revising a game development plan that failed to achieve the desired implementation.

Common game development issues to address:
- Code doesn't follow Unity/Unreal conventions or best practices
- Assets not created in proper project structure
- Missing integration with existing game systems
- Compilation errors or runtime issues
- Incorrect tool usage for game development tasks
- Repeated failures with the same approach

Create a revised development plan that:
- Uses the correct Unity/Unreal development workflow
- Follows established game programming patterns
- Creates properly integrated game features
- Includes adequate testing and validation
- Addresses the specific development failure
- Tries different approaches than what previously failed

VALID TOOL NAMES: search, get_project_info, create_asset, write_file, edit_project_config, get_script_snippets, compile_and_test, scene_management

Focus on professional game development practices and working implementations. Learn from previous failures."""

    # Structure for optimal caching
    messages = [
        {"role": "system", "content": static_repair_content},
        {"role": "user", "content": repair_request}
    ]
    
    try:
        # Use structured output for repair planning with type constraints
        structured_model = model.with_structured_output(StructuredExecutionPlan)
        structured_response = await structured_model.ainvoke(messages)
        
        # Convert to internal format - NO VALIDATION NEEDED!
        steps = []
        for step_data in structured_response.steps:
            step = PlanStep(
                description=step_data.description,
                success_criteria=step_data.success_criteria,
                tool_name=step_data.tool_name,  # Type-safe!
                dependencies=step_data.dependencies
            )
            steps.append(step)
        
        revised_plan = ExecutionPlan(
            goal=state.plan.goal,
            steps=steps
        )
        
        return {
            "plan": revised_plan,
            "step_index": 0,
            "retry_count": 0,  # Reset retry count for new plan
            "current_assessment": None,
            "plan_revision_count": state.plan_revision_count + 1,
            "messages": [AIMessage(content=user_message)]
        }
        
    except Exception as e:
        # Fallback plan
        fallback_steps = create_smart_default_plan(state.plan.goal)
        fallback_plan = ExecutionPlan(
            goal=state.plan.goal,
            steps=fallback_steps
        )
        
        return {
            "plan": fallback_plan,
            "step_index": 0,
            "retry_count": 0,  # Reset retry count
            "current_assessment": None,
            "plan_revision_count": state.plan_revision_count + 1,
            "messages": [AIMessage(content="I'll try a simpler approach to help you.")]
        }