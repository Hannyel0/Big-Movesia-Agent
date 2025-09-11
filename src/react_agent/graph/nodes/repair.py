"""Repair node for the ReAct agent."""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.state import (
    ExecutionPlan,
    PlanStep,
    State,
    ToolName,
)
from react_agent.utils import get_model


# Structured output schemas for repair
class StructuredPlanStep(BaseModel):
    """Structured step for planning output with type-constrained tools."""
    description: str = Field(description="Clear description of what this step accomplishes")
    tool_name: ToolName = Field(description="Specific tool to use for this step")  # Type-constrained!
    success_criteria: str = Field(description="Measurable criteria to determine if step succeeded")
    dependencies: List[int] = Field(default_factory=list, description="Indices of steps this depends on")


class StructuredExecutionPlan(BaseModel):
    """Structured execution plan for native output with type-constrained tools."""
    goal: str = Field(description="Overall goal to achieve")
    steps: List[StructuredPlanStep] = Field(description="Ordered list of steps to execute")


def create_smart_default_plan(user_goal: str) -> List[PlanStep]:
    """Create an intelligent default plan based on the user's goal."""
    # Import here to avoid circular imports
    from react_agent.graph.nodes.plan import create_tool_aware_plan, analyze_user_goal
    
    analysis = analyze_user_goal(user_goal)
    return create_tool_aware_plan(user_goal, analysis)


async def repair(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Replan after encountering issues using structured outputs."""
    context = runtime.context
    model = get_model(context.planning_model or context.model)
    
    if not state.plan:
        return {}
    
    # Create a user-friendly message about trying a different approach
    user_message = "Let me try a different approach to better help you with this."
    
    # Gather information about what went wrong
    current_step = state.plan.steps[state.step_index]
    assessment = state.current_assessment
    
    repair_request = f"""The current execution plan needs adjustment for: "{state.plan.goal}"

Issue: {assessment.reason if assessment else "Unknown issue"}
Failed Step: {current_step.description}
Recommended Tool: {current_step.tool_name}

Create a revised plan that:
1. Addresses the specific issue encountered
2. Uses different tools or approaches
3. Maintains focus on the original goal
4. Has clear, achievable steps

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

Create a revised development plan that:
- Uses the correct Unity/Unreal development workflow
- Follows established game programming patterns
- Creates properly integrated game features
- Includes adequate testing and validation
- Addresses the specific development failure

VALID TOOL NAMES: search, get_project_info, create_asset, write_file, edit_project_config, get_script_snippets, compile_and_test, scene_management

Focus on professional game development practices and working implementations."""

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
            "retry_count": 0,
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
            "retry_count": 0,
            "current_assessment": None,
            "plan_revision_count": state.plan_revision_count + 1,
            "messages": [AIMessage(content="I'll try a simpler approach to help you.")]
        }