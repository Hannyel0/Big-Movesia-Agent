"""Planning node for the ReAct agent."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.prompts import PLANNING_PROMPT
from react_agent.state import (
    ExecutionPlan,
    PlanStep,
    State,
    StepStatus,
    ToolName,
)
from react_agent.tools import TOOLS, TOOL_METADATA
from react_agent.narration import NarrationEngine
from react_agent.utils import get_message_text, get_model


# Build tools description once at module level for caching
def _build_static_tools_description() -> str:
    """Build static tools description for caching."""
    tools_info = []
    for tool in TOOLS:
        tool_info = f"- {tool.name}: {tool.description}"
        if hasattr(tool, 'name') and tool.name in TOOL_METADATA:
            metadata = TOOL_METADATA[tool.name]
            tool_info += f" (Best for: {', '.join(metadata['best_for'])})"
        tools_info.append(tool_info)
    return "\n".join(tools_info)

# Static tools description built once
STATIC_TOOLS_DESCRIPTION = _build_static_tools_description()

# Initialize narration components
narration_engine = NarrationEngine()


# Updated structured output schemas with type constraints
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


def analyze_user_goal(goal: str) -> Dict[str, Any]:
    """Analyze the user's goal to determine required actions and tools."""
    goal_lower = goal.lower()
    
    analysis = {
        "requires_search": False,
        "requires_project_info": False,
        "requires_code_generation": False,
        "requires_asset_creation": False,
        "requires_scene_work": False,
        "requires_compilation": False,
        "requires_config_changes": False,
        "main_action_type": "unknown"
    }
    
    # Determine what the user wants to do
    if any(word in goal_lower for word in ["create script", "write script", "make script", "player controller", "fps controller", "movement script"]):
        analysis.update({
            "requires_search": True,
            "requires_project_info": True,
            "requires_code_generation": True,
            "requires_asset_creation": True,
            "requires_compilation": True,
            "main_action_type": "script_creation"
        })
    
    elif any(word in goal_lower for word in ["create prefab", "make prefab", "build prefab"]):
        analysis.update({
            "requires_project_info": True,
            "requires_asset_creation": True,
            "main_action_type": "prefab_creation"
        })
    
    elif any(word in goal_lower for word in ["create scene", "new scene", "build level", "make level"]):
        analysis.update({
            "requires_project_info": True,
            "requires_scene_work": True,
            "main_action_type": "scene_creation"
        })
    
    elif any(word in goal_lower for word in ["setup", "configure", "settings", "build settings"]):
        analysis.update({
            "requires_project_info": True,
            "requires_config_changes": True,
            "main_action_type": "configuration"
        })
    
    elif any(word in goal_lower for word in ["how to", "tutorial", "learn", "guide", "best practices"]):
        analysis.update({
            "requires_search": True,
            "requires_project_info": True,
            "main_action_type": "tutorial_help"
        })
    
    elif any(word in goal_lower for word in ["debug", "fix", "error", "problem", "issue"]):
        analysis.update({
            "requires_project_info": True,
            "requires_compilation": True,
            "main_action_type": "debugging"
        })
    
    else:
        analysis.update({
            "requires_search": True,
            "requires_project_info": True,
            "main_action_type": "general_help"
        })
    
    return analysis


def create_tool_aware_plan(goal: str, analysis: Dict[str, Any]) -> List[PlanStep]:
    """Create a plan using only available tools based on goal analysis."""
    steps = []
    
    action_type = analysis["main_action_type"]
    
    if action_type == "script_creation":
        if analysis["requires_search"]:
            steps.append(PlanStep(
                description=f"Search for current best practices and tutorials for Unity script development related to: {goal}",
                tool_name="search",  # Type-safe! Must be one of the ToolName literals
                success_criteria="Found relevant, current information about Unity scripting best practices"
            ))
        
        steps.append(PlanStep(
            description="Get current project information to understand the development environment and setup",
            tool_name="get_project_info", 
            success_criteria="Retrieved project structure, Unity version, and installed packages",
            dependencies=[0] if steps else []
        ))
        
        steps.append(PlanStep(
            description="Get appropriate code snippets and templates for the requested script functionality",
            tool_name="get_script_snippets",
            success_criteria="Retrieved relevant code templates that match the requirements",
            dependencies=list(range(len(steps)))
        ))
        
        steps.append(PlanStep(
            description="Write the script file in the project with the generated code",
            tool_name="write_file",
            success_criteria="Successfully created and wrote the script file to the project",
            dependencies=list(range(len(steps)))
        ))
        
        steps.append(PlanStep(
            description="Compile and test the script to ensure it works correctly",
            tool_name="compile_and_test",
            success_criteria="Script compiles without errors and integrates properly",
            dependencies=list(range(len(steps)))
        ))
    
    elif action_type == "prefab_creation":
        steps.append(PlanStep(
            description="Get current project information to understand available resources",
            tool_name="get_project_info",
            success_criteria="Retrieved project structure and available assets"
        ))
        
        steps.append(PlanStep(
            description=f"Create the requested prefab asset: {goal}",
            tool_name="create_asset",
            success_criteria="Successfully created prefab with appropriate components",
            dependencies=[0]
        ))
    
    elif action_type == "scene_creation":
        steps.append(PlanStep(
            description="Get project information to understand current scene structure",
            tool_name="get_project_info",
            success_criteria="Retrieved current project and scene information"
        ))
        
        steps.append(PlanStep(
            description=f"Create and set up the new scene: {goal}",
            tool_name="scene_management",
            success_criteria="Successfully created new scene with basic setup",
            dependencies=[0]
        ))
    
    elif action_type == "configuration":
        steps.append(PlanStep(
            description="Get current project configuration to understand what needs to be changed",
            tool_name="get_project_info",
            success_criteria="Retrieved current project settings and configuration"
        ))
        
        steps.append(PlanStep(
            description=f"Apply the requested configuration changes: {goal}",
            tool_name="edit_project_config",
            success_criteria="Successfully updated project configuration settings",
            dependencies=[0]
        ))
        
        steps.append(PlanStep(
            description="Test the configuration changes by compiling the project",
            tool_name="compile_and_test",
            success_criteria="Project compiles successfully with new configuration",
            dependencies=[1]
        ))
    
    elif action_type == "tutorial_help":
        steps.append(PlanStep(
            description=f"Search for current tutorials and documentation about: {goal}",
            tool_name="search",
            success_criteria="Found comprehensive, up-to-date tutorial information"
        ))
        
        steps.append(PlanStep(
            description="Get project information to provide context-specific guidance",
            tool_name="get_project_info",
            success_criteria="Retrieved project details to tailor advice appropriately",
            dependencies=[0]
        ))
        
        if any(word in goal.lower() for word in ["script", "code", "programming", "controller", "system"]):
            steps.append(PlanStep(
                description="Get relevant code examples and snippets for hands-on learning",
                tool_name="get_script_snippets",
                success_criteria="Retrieved practical code examples for the tutorial topic",
                dependencies=list(range(len(steps)))
            ))
    
    elif action_type == "debugging":
        steps.append(PlanStep(
            description="Get current project information to understand the problem context",
            tool_name="get_project_info",
            success_criteria="Retrieved project structure and current state information"
        ))
        
        steps.append(PlanStep(
            description="Compile and test the project to identify specific issues",
            tool_name="compile_and_test",
            success_criteria="Identified compilation errors, warnings, or runtime issues",
            dependencies=[0]
        ))
        
        steps.append(PlanStep(
            description="Search for solutions to the identified problems",
            tool_name="search",
            success_criteria="Found relevant troubleshooting information and solutions",
            dependencies=[1]
        ))
    
    else:  # general_help
        steps.append(PlanStep(
            description=f"Search for information and guidance about: {goal}",
            tool_name="search",
            success_criteria="Found relevant information about the requested topic"
        ))
        
        steps.append(PlanStep(
            description="Get project context to provide personalized advice",
            tool_name="get_project_info",
            success_criteria="Retrieved project information to give specific guidance",
            dependencies=[0]
        ))
    
    return steps


def create_smart_default_plan(user_goal: str) -> List[PlanStep]:
    """Create an intelligent default plan based on the user's goal."""
    analysis = analyze_user_goal(user_goal)
    return create_tool_aware_plan(user_goal, analysis)


async def plan(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Enhanced planning with rich introductory narration."""
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
        return {"messages": [AIMessage(content="I didn't receive a clear request. Could you please let me know what you'd like help with?")]}
    
    # Analyze the goal to understand what tools we'll need
    analysis = analyze_user_goal(user_message)
    
    # Create action guidance
    action_guidance = ""
    if analysis["main_action_type"] == "script_creation":
        action_guidance = """
For script creation tasks:
1. Start with 'search' to find current best practices
2. Use 'get_project_info' to understand the Unity setup
3. Use 'get_script_snippets' to get code templates
4. Use 'write_file' to create the actual script
5. Use 'compile_and_test' to verify it works"""
    
    elif analysis["main_action_type"] == "tutorial_help":
        action_guidance = """
For tutorial/learning tasks:
1. Use 'search' to find current tutorials and documentation
2. Use 'get_project_info' to provide context-specific advice
3. Optionally use 'get_script_snippets' for code examples"""
    
    # Enhanced planning prompt for structured output
    planning_request = f"""Create a tactical Unity/game development plan for: "{user_message}"

VALID TOOL NAMES: search, get_project_info, create_asset, write_file, edit_project_config, get_script_snippets, compile_and_test, scene_management

{action_guidance}

Create a plan that uses ONLY the available tools above. Every step MUST specify a valid tool_name from the list.
Focus on 2-5 logical steps that solve the user's request effectively."""
    
    # Structure messages for optimal caching - static content first
    # System prompt with tools info - this will be cached
    static_system_content = f"""{PLANNING_PROMPT}

Your available game development tools:
{STATIC_TOOLS_DESCRIPTION}

IMPORTANT: Every step must use a specific tool from your toolkit. No generic or non-executable steps.

Plan Structure Guidelines:
- Start with research (search) for best practices and current approaches
- Gather project context (get_project_info) to understand the current setup
- Use code generation tools (get_script_snippets) for implementation details
- Create/modify assets (create_asset, write_file) for concrete deliverables
- Test implementations (compile_and_test) to ensure quality
- Configure settings (edit_project_config) when needed
- Manage scenes (scene_management) for level/world changes

Create plans that result in working game features, not just documentation."""

    # Construct messages with static content first for caching
    messages = [
        {"role": "system", "content": static_system_content},
        {"role": "user", "content": planning_request}
    ]
    
    try:
        # Use structured output to force proper JSON schema with type constraints
        structured_model = model.with_structured_output(StructuredExecutionPlan)
        structured_response = await structured_model.ainvoke(messages)
        
        # Convert structured response to internal format
        # NO MORE VALIDATION NEEDED! Type constraints guarantee valid tool names
        steps = []
        for step_data in structured_response.steps:
            step = PlanStep(
                description=step_data.description,
                success_criteria=step_data.success_criteria,
                tool_name=step_data.tool_name,  # Already type-safe!
                dependencies=step_data.dependencies
            )
            steps.append(step)
        
        plan = ExecutionPlan(
            goal=structured_response.goal,
            steps=steps
        )
        
        # CREATE RICH PLANNING NARRATION
        planning_narration = narration_engine.create_planning_narration(plan, user_message)
        
        # Add streaming UI updates if supported
        if context.runtime_metadata.get("supports_streaming"):
            # Import here to avoid circular imports
            from react_agent.narration import StreamingNarrator
            streaming_narrator = StreamingNarrator()
            
            for i, step in enumerate(plan.steps):
                ui_update = streaming_narrator.create_inline_update(
                    f"📌 Step {i+1}: {step.description}",
                    style="info"
                )
                if hasattr(runtime, 'push_ui_message'):
                    runtime.push_ui_message(ui_update)
        
        return {
            "plan": plan,
            "step_index": 0,
            "retry_count": 0,
            "messages": [AIMessage(content=planning_narration)]
        }
    
    except Exception as e:
        # Fallback to analysis-based planning if structured output fails
        fallback_steps = create_tool_aware_plan(user_message, analysis)
        fallback_plan = ExecutionPlan(
            goal=user_message,
            steps=fallback_steps
        )
        
        fallback_narration = (f"I'll help you with **{user_message}**.\n\n"
                             f"Let me work through this systematically using my Unity development tools.")
        
        return {
            "plan": fallback_plan,
            "step_index": 0,
            "retry_count": 0,
            "messages": [AIMessage(content=fallback_narration)]
        }