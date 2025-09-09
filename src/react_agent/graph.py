"""Enhanced ReAct agent with type-constrained tool planning."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Literal, List

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.prompts import (
    FINAL_SUMMARY_PROMPT,
    PLANNING_PROMPT,
)
from react_agent.state import (
    AssessmentOutcome,
    ExecutionPlan,
    InputState,
    PlanStep,
    State,
    StepStatus,
    ToolName,  # Import the type-constrained tool name
)
from react_agent.tools import TOOLS, TOOL_METADATA
from react_agent.narration import NarrationEngine, StreamingNarrator, integrate_narration_engine
from react_agent.utils import (
    get_message_text,
    get_model,
    create_dynamic_step_message,
    create_varied_post_tool_message,
)


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
streaming_narrator = StreamingNarrator()


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


class StructuredAssessment(BaseModel):
    """Structured assessment for native output."""
    outcome: Literal["success", "retry", "blocked"] = Field(
        description="Whether the step succeeded, should be retried, or is blocked"
    )
    reason: str = Field(description="Specific explanation of the assessment")
    fix: str = Field(default="", description="Suggested fix if retry is needed")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence in assessment")


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


# Core node functions

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


async def act(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Enhanced act node with rich narration from tool outputs."""
    context = runtime.context
    model = get_model(context.model)
    
    if not state.plan or state.step_index >= len(state.plan.steps):
        return {"messages": [AIMessage(content="I'm having trouble processing your request. Could you please try rephrasing it?")]}
    
    current_step = state.plan.steps[state.step_index]
    
    # START STREAMING NARRATION (if UI supports it)
    stream_msg = None
    if context.runtime_metadata.get("supports_streaming"):
        stream_msg = streaming_narrator.start_step_narration(
            f"Step {state.step_index + 1}: {current_step.description}"
        )
        # Push to UI stream if available
        if hasattr(runtime, 'push_ui_message'):
            runtime.push_ui_message(stream_msg)
    
    # Create rich pre-step narration using the engine
    step_context = {
        "step_index": state.step_index,
        "total_steps": len(state.plan.steps),
        "goal": state.plan.goal,
        "tool_name": current_step.tool_name,
        "description": current_step.description
    }
    
    # Generate contextual pre-step message
    pre_step_narration = _create_rich_pre_step_narration(current_step, step_context)
    
    # Update streaming if active
    if stream_msg and context.runtime_metadata.get("supports_streaming"):
        updated_msg = streaming_narrator.update_step_progress(
            stream_msg["id"], 
            "Executing tool call...",
            f"Using {current_step.tool_name}"
        )
        if hasattr(runtime, 'push_ui_message'):
            runtime.push_ui_message(updated_msg)
    
    # Prepare execution context
    execution_context = {
        "current_step": current_step.description,
        "success_criteria": current_step.success_criteria,
        "step_number": state.step_index + 1,
        "total_steps": len(state.plan.steps),
        "required_tool": current_step.tool_name,
        "available_tools": [tool.name for tool in TOOLS]
    }
    
    # Create focused action prompt that mandates tool usage
    action_request = f"""You are executing step {state.step_index + 1} of {len(state.plan.steps)} for: "{state.plan.goal}"

CURRENT STEP: {current_step.description}
SUCCESS CRITERIA: {current_step.success_criteria}
REQUIRED TOOL: {current_step.tool_name}

YOU MUST USE THE REQUIRED TOOL: {current_step.tool_name}

Based on the step description and required tool, call {current_step.tool_name} with appropriate parameters to complete this step."""
    
    # Get conversation history for context
    conversation_messages = []
    for msg in state.messages[-3:]:
        if isinstance(msg, HumanMessage):
            conversation_messages.append({"role": "user", "content": get_message_text(msg)})
        elif isinstance(msg, AIMessage) and not msg.tool_calls:
            conversation_messages.append({"role": "assistant", "content": get_message_text(msg)})
    
    # Static system content that can be cached
    static_system_content = """You are executing a Unity/Unreal Engine development step. Focus on creating working game features using your development tools.

DEVELOPMENT EXECUTION GUIDELINES:
1. Use the specified tool to create actual game development deliverables
2. Follow Unity/Unreal best practices and naming conventions
3. Write production-ready code that integrates with existing projects
4. Create assets that follow proper game development workflows
5. Focus on this specific development task completely

Tool-Specific Execution:
- search: Find current Unity/Unreal tutorials, best practices, and solutions
- get_project_info: Analyze project structure, version, packages, and setup
- get_script_snippets: Retrieve working code templates for game systems
- create_asset: Make new scripts, prefabs, materials, or scenes
- write_file: Create actual C# scripts or configuration files
- scene_management: Build levels, set up gameplay areas, place objects
- compile_and_test: Verify code compiles and features work correctly
- edit_project_config: Modify build settings, input, quality, or player settings

EXECUTE THE DEVELOPMENT STEP NOW - create working game development output."""

    # Dynamic context message - this part changes and won't be cached
    dynamic_context_message = f"""Current execution context:
{json.dumps(execution_context, indent=2)}"""

    # Structure for optimal caching
    messages = [
        {"role": "system", "content": static_system_content},
        {"role": "system", "content": dynamic_context_message},
        *conversation_messages,
        {"role": "user", "content": action_request}
    ]
    
    try:
        # Bind tools to model and invoke with forced tool choice
        model_with_tools = model.bind_tools(TOOLS)
        
        # Force the specific tool usage
        response = await model_with_tools.ainvoke(
            messages,
            tool_choice={"type": "function", "function": {"name": current_step.tool_name}}
        )
        
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
        
        # Create messages that include BOTH the user update AND the tool call
        messages_to_add = [
            AIMessage(content=pre_step_narration),  # Rich contextual narration
            response  # Tool call response
        ]
        
        return {
            "plan": updated_plan,
            "messages": messages_to_add,
            "total_tool_calls": state.total_tool_calls + (len(response.tool_calls) if response.tool_calls else 0)
        }
        
    except Exception as e:
        # Rich error narration
        error_narration = narration_engine._narrate_error(
            f"step {state.step_index + 1}", 
            str(e)
        )
        
        # Complete streaming with error if active
        if stream_msg and context.runtime_metadata.get("supports_streaming"):
            streaming_narrator.complete_step(stream_msg["id"], f"Error: {str(e)}")
        
        updated_steps = list(state.plan.steps)
        error_messages = current_step.error_messages + [str(e)]
        updated_steps[state.step_index] = PlanStep(
            description=current_step.description,
            tool_name=current_step.tool_name,
            success_criteria=current_step.success_criteria,
            dependencies=current_step.dependencies,
            status=current_step.status,
            attempts=current_step.attempts + 1,
            error_messages=error_messages
        )
        
        updated_plan = ExecutionPlan(
            goal=state.plan.goal,
            steps=updated_steps,
            metadata=state.plan.metadata
        )
        
        return {
            "plan": updated_plan,
            "messages": [AIMessage(content=error_narration)],
            "tool_errors": state.tool_errors + [{"step": state.step_index, "error": str(e)}]
        }


async def assess(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Enhanced assessment with rich post-tool narration."""
    context = runtime.context
    model = get_model(context.assessment_model or context.model)
    
    if not state.plan or state.step_index >= len(state.plan.steps):
        return {}
    
    current_step = state.plan.steps[state.step_index]
    
    # Extract the tool result from the last ToolMessage
    tool_result = None
    tool_name = current_step.tool_name
    
    for msg in reversed(state.messages[-10:]):  # Look at recent messages
        if isinstance(msg, ToolMessage):
            try:
                # Parse tool result
                content = get_message_text(msg)
                tool_result = json.loads(content) if content else {}
            except json.JSONDecodeError:
                tool_result = {"message": content}
            break
    
    # GENERATE RICH POST-TOOL NARRATION
    post_tool_narration = None
    if tool_result and tool_name:
        step_context = {
            "step_index": state.step_index,
            "total_steps": len(state.plan.steps),
            "goal": state.plan.goal,
            "tool_name": tool_name,
            "description": current_step.description
        }
        
        # Use the narration engine to create rich, contextual narration
        post_tool_narration = narration_engine.narrate_tool_result(
            tool_name,
            tool_result,
            step_context
        )
    
    # Find the most recent tool messages for better context
    recent_tool_results = []
    latest_tool_result = None
    for msg in reversed(state.messages):
        if isinstance(msg, ToolMessage):
            if latest_tool_result is None:
                try:
                    latest_tool_result = json.loads(get_message_text(msg))
                except:
                    latest_tool_result = {"message": get_message_text(msg)}
            recent_tool_results.append(f"Tool '{msg.name}': {get_message_text(msg)}")
            if len(recent_tool_results) >= 3:
                break
    
    tool_results_text = "\n".join(recent_tool_results) if recent_tool_results else "No recent tool results found"
    
    # Now perform the assessment using structured output
    assessment_request = f"""Assess if the current step has been successfully completed:

Overall Goal: {state.plan.goal}
Current Step ({state.step_index + 1}/{len(state.plan.steps)}): {current_step.description}
Success Criteria: {current_step.success_criteria}
Tool Used: {tool_name}

Tool Result Summary: {json.dumps(tool_result, indent=2) if tool_result else "No result captured"}

Determine if the step succeeded, needs retry, or is blocked."""
    
    # Static assessment prompt - cacheable
    static_assessment_content = """You are evaluating game development step completion with focus on deliverable quality and tool effectiveness.

Assessment Criteria:
- Was the required development tool used properly?
- Did the step produce a working game asset, script, or feature?
- Can the output be integrated into a Unity/Unreal project?
- Does the result follow game development best practices?
- Is there sufficient progress toward the gameplay goal?

Be particularly strict about:
- Code quality and Unity/Unreal compatibility
- Asset creation and proper file structure
- Project integration and build compatibility
- Following established game development patterns

Assessment outcomes:
- "success": Step completed with working game development output
- "retry": Implementation incomplete or doesn't meet game dev standards
- "blocked": Technical limitation preventing proper implementation

Judge based on professional game development quality and deliverables."""

    # Structure for optimal caching
    messages = [
        {"role": "system", "content": static_assessment_content},
        {"role": "user", "content": assessment_request}
    ]
    
    try:
        # Use structured output for assessment
        structured_model = model.with_structured_output(StructuredAssessment)
        structured_assessment = await structured_model.ainvoke(messages)
        
        assessment = AssessmentOutcome(
            outcome=structured_assessment.outcome,
            reason=structured_assessment.reason,
            fix=structured_assessment.fix or None,
            confidence=structured_assessment.confidence
        )
        
        # Prepare messages with rich narration
        messages_to_add = []
        if post_tool_narration:
            messages_to_add.append(AIMessage(content=post_tool_narration))
        
        # Add transition narration if moving to next step
        if assessment.outcome == "success" and state.step_index + 1 < len(state.plan.steps):
            next_step = state.plan.steps[state.step_index + 1]
            transition = _create_step_transition(current_step, next_step)
            if transition:
                messages_to_add.append(AIMessage(content=transition))
        
        result = {
            "current_assessment": assessment,
            "total_assessments": state.total_assessments + 1
        }
        
        if messages_to_add:
            result["messages"] = messages_to_add
        
        # Complete streaming narration if active
        if context.runtime_metadata.get("supports_streaming"):
            stream_id = f"step_{state.step_index}"
            if assessment.outcome == "success":
                streaming_narrator.complete_step(stream_id, "Step completed successfully!")
            elif assessment.outcome == "retry":
                streaming_narrator.update_step_progress(stream_id, "Retrying with adjustments...")
            else:
                streaming_narrator.complete_step(stream_id, "Step blocked - replanning...")
        
        return result
        
    except Exception as e:
        # Fallback assessment with narration
        fallback_narration = f"Assessment check completed. Moving forward with the implementation."
        
        return {
            "current_assessment": AssessmentOutcome(
                outcome="success" if tool_result else "retry",
                reason="Automated assessment",
                confidence=0.5
            ),
            "messages": [AIMessage(content=fallback_narration)] if tool_result else [],
            "total_assessments": state.total_assessments + 1
        }


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


async def finish(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Enhanced finish node with rich completion summary."""
    context = runtime.context
    
    # Use narration engine for rich completion summary
    completion_narration = narration_engine.create_completion_narration(
        state.completed_steps,
        state.plan
    )
    
    # Add final streaming updates if supported
    if context.runtime_metadata.get("supports_streaming"):
        final_update = streaming_narrator.create_inline_update(
            "✨ Implementation complete! Check your Unity project for the new features.",
            style="success"
        )
        if hasattr(runtime, 'push_ui_message'):
            runtime.push_ui_message(final_update)
    
    return {
        "messages": [AIMessage(content=completion_narration)]
    }


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
        "retry_count": 0,
        "current_assessment": None
    }
    
    if progress_message:
        result["messages"] = [progress_message]
    
    return result


# Routing functions
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

# Helper functions for rich narration
def _create_rich_pre_step_narration(step: 'PlanStep', context: Dict[str, Any]) -> str:
    """Create rich, contextual pre-step narration."""
    step_num = context.get("step_index", 0) + 1
    total_steps = context.get("total_steps", 1)
    
    # Tool-specific introductions with variety
    tool_intros = {
        "search": [
            f"**Step {step_num}/{total_steps}**: Researching current Unity best practices and tutorials...\n\nI'm looking for the most up-to-date and authoritative sources to ensure we're following modern game development patterns.",
            f"**Researching ({step_num}/{total_steps})**: Let me find the latest Unity documentation and community solutions for this specific implementation.",
            f"**Step {step_num}**: Diving into Unity resources to find proven approaches for your request..."
        ],
        "get_project_info": [
            f"**Step {step_num}/{total_steps}**: Analyzing your project structure...\n\nI'm examining your Unity setup, installed packages, and current configuration to tailor the solution perfectly to your environment.",
            f"**Project Analysis ({step_num}/{total_steps})**: Scanning your project to understand the existing setup and dependencies.",
            f"**Step {step_num}**: Inspecting your Unity project to ensure compatibility..."
        ],
        "write_file": [
            f"**Step {step_num}/{total_steps}**: Creating your script...\n\nI'm writing production-ready C# code that follows Unity conventions and integrates seamlessly with your project.",
            f"**Code Generation ({step_num}/{total_steps})**: Building the script with proper namespaces, optimized logic, and clear documentation.",
            f"**Step {step_num}**: Writing the implementation to your project..."
        ],
        "compile_and_test": [
            f"**Step {step_num}/{total_steps}**: Running build validation...\n\nI'm compiling the project to ensure everything integrates properly and checking for any issues.",
            f"**Testing ({step_num}/{total_steps})**: Verifying that the new code compiles cleanly and works with your existing systems.",
            f"**Step {step_num}**: Building the project to validate the implementation..."
        ]
    }
    
    # Get varied intro for this tool
    tool_name = step.tool_name or "process"
    intros = tool_intros.get(tool_name, [f"**Step {step_num}/{total_steps}**: {step.description}"])
    
    # Select with variety
    import hashlib
    hash_val = int(hashlib.md5(f"{step_num}_{tool_name}".encode()).hexdigest(), 16)
    return intros[hash_val % len(intros)]


def _create_step_transition(current_step: 'PlanStep', next_step: 'PlanStep') -> str:
    """Create natural transition narration between steps."""
    transitions = [
        f"Excellent! Now that {current_step.tool_name} is complete, let's move on to {next_step.description.lower()}.",
        f"Perfect. With that done, the next step is to {next_step.description.lower()}.",
        f"Great progress! Moving forward to {next_step.description.lower()}.",
        ""  # Sometimes no transition for flow
    ]
    
    import hashlib
    hash_val = int(hashlib.md5(f"{current_step.tool_name}_{next_step.tool_name}".encode()).hexdigest(), 16)
    return transitions[hash_val % len(transitions)]


# Export the compiled graph
graph = create_graph()