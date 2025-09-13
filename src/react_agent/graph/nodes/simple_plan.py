"""Simple planning node for straightforward requests requiring minimal steps."""

from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.state import (
    ExecutionPlan,
    PlanStep,
    State,
    StructuredExecutionPlan,
)
from react_agent.utils import get_message_text, get_model


async def simple_plan(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Create a focused 2-3 step plan for straightforward development tasks."""
    context = runtime.context
    model = get_model(context.planning_model or context.model)
    
    # Extract user request
    user_request = None
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            user_request = get_message_text(msg)
            break
    
    if not user_request:
        return {"messages": [AIMessage(content="I need a clear request to create a plan.")]}
    
    # Get complexity reasoning from previous classification
    complexity_reasoning = state.runtime_metadata.get(
        "complexity_reasoning", 
        "Straightforward task requiring focused approach"
    )
    
    # Simple planning prompt - focused on efficiency
    simple_planning_request = f"""Create a focused, efficient plan for this Unity/game development task: "{user_request}"

Complexity Assessment: {complexity_reasoning}

CONSTRAINTS:
- Maximum 3 steps total
- Each step must use a specific tool from: search, get_project_info, create_asset, write_file, edit_project_config, get_script_snippets, compile_and_test, scene_management
- Focus on the most essential actions only
- Eliminate unnecessary research or verification steps for straightforward tasks
- Prioritize direct implementation over extensive preparation

APPROACH:
- If it's a creation task: go straight to implementation (skip extensive research)
- If it's a fix/debug task: identify → fix → verify
- If it's a configuration task: check current state → apply changes
- If it's an information task: get relevant info → provide context

Create a lean, actionable plan that gets to the solution quickly."""

    # Static simple planning system content
    static_simple_system_content = """You are creating efficient, focused development plans for straightforward Unity/Unreal tasks.

PLANNING PRINCIPLES:
- Favor direct action over extensive research
- Use 2-3 steps maximum
- Skip redundant verification unless critical
- Combine related operations when possible
- Assume standard Unity best practices unless otherwise specified

VALID TOOLS: search, get_project_info, create_asset, write_file, edit_project_config, get_script_snippets, compile_and_test, scene_management

EFFICIENCY GUIDELINES:
- For script creation: get_script_snippets → write_file (→ compile_and_test if complex)
- For asset creation: get_project_info → create_asset
- For configuration: get_project_info → edit_project_config
- For debugging: get_project_info → compile_and_test
- For information: search (→ get_project_info if context needed)

Create focused plans that solve the user's request efficiently."""

    # Structure messages for optimal caching
    messages = [
        {"role": "system", "content": static_simple_system_content},
        {"role": "user", "content": simple_planning_request}
    ]
    
    try:
        # Use structured output for reliable planning
        structured_model = model.with_structured_output(StructuredExecutionPlan)
        structured_response = await structured_model.ainvoke(messages)
        
        # Enforce maximum step limit
        limited_steps = structured_response.steps[:3]  # Hard limit to 3 steps
        
        # Convert to internal format
        steps = []
        for step_data in limited_steps:
            step = PlanStep(
                description=step_data.description,
                success_criteria=step_data.success_criteria,
                tool_name=step_data.tool_name,
                dependencies=step_data.dependencies
            )
            steps.append(step)
        
        plan = ExecutionPlan(
            goal=structured_response.goal,
            steps=steps,
            metadata={"planning_mode": "simple", "original_step_count": len(structured_response.steps)}
        )
        
        # Create streamlined narration for simple plans
        step_count = len(steps)
        narration = f"I'll handle this efficiently with {step_count} focused step{'s' if step_count != 1 else ''}:"
        
        for i, step in enumerate(steps, 1):
            narration += f"\n{i}. {step.description}"
        
        return {
            "plan": plan,
            "step_index": 0,
            "retry_count": 0,
            "messages": [AIMessage(content=narration)]
        }
        
    except Exception as e:
        # Fallback to minimal hardcoded plan
        fallback_steps = _create_minimal_fallback_plan(user_request)
        fallback_plan = ExecutionPlan(
            goal=user_request,
            steps=fallback_steps,
            metadata={"planning_mode": "simple_fallback"}
        )
        
        fallback_narration = "I'll take a direct approach to help you with this."
        
        return {
            "plan": fallback_plan,
            "step_index": 0,
            "retry_count": 0,
            "messages": [AIMessage(content=fallback_narration)]
        }


def _create_minimal_fallback_plan(user_request: str) -> list[PlanStep]:
    """Create a minimal fallback plan when structured planning fails."""
    request_lower = user_request.lower()
    
    # Simple pattern matching for common requests
    if any(word in request_lower for word in ["create script", "write script", "make script"]):
        return [
            PlanStep(
                description="Get code templates for the requested script",
                tool_name="get_script_snippets",
                success_criteria="Retrieved appropriate code templates"
            ),
            PlanStep(
                description="Create the script file with the generated code",
                tool_name="write_file",
                success_criteria="Successfully wrote script file to project",
                dependencies=[0]
            )
        ]
    
    elif any(word in request_lower for word in ["project info", "project status", "what's in"]):
        return [
            PlanStep(
                description="Retrieve current project information and structure",
                tool_name="get_project_info",
                success_criteria="Successfully retrieved project details"
            )
        ]
    
    elif any(word in request_lower for word in ["compile", "test", "build"]):
        return [
            PlanStep(
                description="Compile and test the current project",
                tool_name="compile_and_test",
                success_criteria="Successfully compiled and tested project"
            )
        ]
    
    elif any(word in request_lower for word in ["create", "new", "make"]):
        return [
            PlanStep(
                description="Get project context for asset creation",
                tool_name="get_project_info",
                success_criteria="Retrieved project structure and context"
            ),
            PlanStep(
                description="Create the requested asset",
                tool_name="create_asset",
                success_criteria="Successfully created the requested asset",
                dependencies=[0]
            )
        ]
    
    else:
        # Generic fallback
        return [
            PlanStep(
                description="Search for information about the requested topic",
                tool_name="search",
                success_criteria="Found relevant information and guidance"
            ),
            PlanStep(
                description="Get project context to provide specific advice",
                tool_name="get_project_info",
                success_criteria="Retrieved project information for context",
                dependencies=[0]
            )
        ]