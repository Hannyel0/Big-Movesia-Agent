"""Enhanced ReAct agent with tool-aware planning that creates achievable steps."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Literal, List

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
from react_agent.tools import TOOLS, TOOL_METADATA
from react_agent.utils import get_message_text, get_model


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
            "requires_search": True,  # Research best practices
            "requires_project_info": True,  # Understand current setup
            "requires_code_generation": True,  # Get code snippets
            "requires_asset_creation": True,  # Create the script file
            "requires_compilation": True,  # Test the script
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
            "requires_project_info": True,  # To give context-specific advice
            "main_action_type": "tutorial_help"
        })
    
    elif any(word in goal_lower for word in ["debug", "fix", "error", "problem", "issue"]):
        analysis.update({
            "requires_project_info": True,
            "requires_compilation": True,
            "main_action_type": "debugging"
        })
    
    else:
        # Default: assume they want general help
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
        # Multi-step plan for script creation
        if analysis["requires_search"]:
            steps.append(PlanStep(
                description=f"Research current best practices and tutorials for Unity script development related to: {goal}",
                tool_name="search",
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
            description="Create the script file in the project with the generated code",
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
        
        # If the tutorial might involve code, get relevant snippets
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
    
    # Ensure every step has a valid tool
    for step in steps:
        if step.tool_name is None:
            # This should never happen with our new approach, but as a safety net
            step.tool_name = "get_project_info"  # Safe fallback tool
    
    return steps


def parse_plan_from_response(response_text: str, user_goal: str) -> ExecutionPlan:
    """Parse a structured plan from LLM response text, ensuring all steps have tools."""
    
    # Try to extract JSON plan first
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        try:
            plan_data = json.loads(json_match.group())
            if 'steps' in plan_data:
                steps = []
                for i, step_data in enumerate(plan_data['steps']):
                    tool_name = step_data.get('tool_name')
                    
                    # Validate that the tool exists
                    available_tools = [tool.name for tool in TOOLS]
                    if tool_name not in available_tools:
                        # Try to map to a valid tool based on step description
                        description_lower = step_data.get('description', '').lower()
                        if any(word in description_lower for word in ['search', 'research', 'find', 'look up']):
                            tool_name = 'search'
                        elif any(word in description_lower for word in ['project', 'info', 'structure', 'setup']):
                            tool_name = 'get_project_info'
                        elif any(word in description_lower for word in ['code', 'script', 'snippet']):
                            tool_name = 'get_script_snippets'
                        elif any(word in description_lower for word in ['create', 'make', 'build']):
                            tool_name = 'create_asset'
                        elif any(word in description_lower for word in ['write', 'file']):
                            tool_name = 'write_file'
                        elif any(word in description_lower for word in ['scene', 'level']):
                            tool_name = 'scene_management'
                        elif any(word in description_lower for word in ['compile', 'test']):
                            tool_name = 'compile_and_test'
                        elif any(word in description_lower for word in ['config', 'setting']):
                            tool_name = 'edit_project_config'
                        else:
                            tool_name = 'get_project_info'  # Safe fallback
                    
                    step = PlanStep(
                        description=step_data.get('description', f'Execute step {i+1}'),
                        success_criteria=step_data.get('success_criteria', 'Step completed successfully'),
                        tool_name=tool_name,
                        dependencies=step_data.get('dependencies', [])
                    )
                    steps.append(step)
                
                if steps:  # Only return if we have valid steps
                    return ExecutionPlan(
                        goal=user_goal,
                        steps=steps,
                        metadata=plan_data.get('metadata', {})
                    )
        except json.JSONDecodeError:
            pass
    
    # If JSON parsing failed, fall back to our analysis-based planning
    analysis = analyze_user_goal(user_goal)
    steps = create_tool_aware_plan(user_goal, analysis)
    
    return ExecutionPlan(
        goal=user_goal,
        steps=steps
    )


def create_smart_default_plan(user_goal: str) -> List[PlanStep]:
    """Create an intelligent default plan based on the user's goal - fallback for game development."""
    
    # Use our analysis-based planning as the smart default
    analysis = analyze_user_goal(user_goal)
    return create_tool_aware_plan(user_goal, analysis)


def create_user_friendly_step_message(step: PlanStep, step_num: int, total_steps: int) -> str:
    """Create a user-friendly message explaining what the agent is doing."""
    
    # Map tool names to base form actions (infinitive) with "Now I'll" prefix
    tool_actions = {
        "search": "search for",
        "get_project_info": "check your project setup",
        "get_script_snippets": "get code templates for",
        "create_asset": "create",
        "write_file": "write",
        "scene_management": "set up scene for",
        "compile_and_test": "test and compile",
        "edit_project_config": "configure project settings for"
    }
    
    action = tool_actions.get(step.tool_name, "work on")
    
    # Create step-specific messages with "Now I'll" prefix
    if step.tool_name == "search":
        message = f"**Step {step_num}/{total_steps}**: Now I'll {action} the latest Unity best practices and tutorials..."
    elif step.tool_name == "get_project_info":
        message = f"**Step {step_num}/{total_steps}**: Now I'll {action} to understand your current Unity environment..."
    elif step.tool_name == "get_script_snippets":
        message = f"**Step {step_num}/{total_steps}**: Now I'll {action} your script implementation..."
    elif step.tool_name == "create_asset":
        message = f"**Step {step_num}/{total_steps}**: Now I'll {action} the requested asset in your project..."
    elif step.tool_name == "write_file":
        message = f"**Step {step_num}/{total_steps}**: Now I'll {action} the script file to your project..."
    elif step.tool_name == "scene_management":
        message = f"**Step {step_num}/{total_steps}**: Now I'll {action} your request..."
    elif step.tool_name == "compile_and_test":
        message = f"**Step {step_num}/{total_steps}**: Now I'll {action} your project to ensure everything works..."
    elif step.tool_name == "edit_project_config":
        message = f"**Step {step_num}/{total_steps}**: Now I'll {action} your request..."
    else:
        message = f"**Step {step_num}/{total_steps}**: Now I'll {action} {step.description.lower()}..."
    
    return message


# Core node functions

async def plan(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Create a comprehensive execution plan that uses available tools effectively."""
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
    
    # Prepare detailed tools information for planning
    tools_info = []
    for tool in TOOLS:
        tool_info = f"- {tool.name}: {tool.description}"
        # Add usage context from tool metadata
        if hasattr(tool, 'name') and tool.name in TOOL_METADATA:
            metadata = TOOL_METADATA[tool.name]
            tool_info += f" (Best for: {', '.join(metadata['best_for'])})"
        tools_info.append(tool_info)
    
    tools_description = "\n".join(tools_info)
    
    # Create a targeted planning prompt based on the goal analysis
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
    
    # Enhanced planning prompt that emphasizes tool usage
    planning_request = f"""You are helping with this Unity/game development request: "{user_message}"

AVAILABLE TOOLS (you MUST use these - no null tool_name values):
{tools_description}

{action_guidance}

Create a JSON plan that uses ONLY the available tools above. Every step MUST specify a valid tool_name.

Example format:
{{
    "goal": "{user_message}",
    "steps": [
        {{
            "description": "Research Unity best practices for player controllers",
            "tool_name": "search",
            "success_criteria": "Found current Unity controller patterns and tutorials",
            "dependencies": []
        }},
        {{
            "description": "Get current project setup information",
            "tool_name": "get_project_info", 
            "success_criteria": "Retrieved Unity version, packages, and project structure",
            "dependencies": [0]
        }},
        {{
            "description": "Get movement controller code templates",
            "tool_name": "get_script_snippets",
            "success_criteria": "Retrieved appropriate movement script templates",
            "dependencies": [1]
        }},
        {{
            "description": "Create the PlayerController.cs script file",
            "tool_name": "write_file",
            "success_criteria": "Successfully wrote the script to Assets/Scripts/",
            "dependencies": [2]
        }},
        {{
            "description": "Compile and test the new script",
            "tool_name": "compile_and_test",
            "success_criteria": "Script compiles without errors",
            "dependencies": [3]
        }}
    ]
}}

CRITICAL: Every step must have a tool_name from the available tools. Create 2-5 logical steps that solve the user's request."""
    
    messages = [
        {"role": "system", "content": PLANNING_PROMPT.format(tools_info=tools_description)},
        {"role": "user", "content": planning_request}
    ]
    
    try:
        response = await model.ainvoke(messages)
        plan_content = get_message_text(response)
        
        # Parse the structured plan from the response
        plan = parse_plan_from_response(plan_content, user_message)
        
        # Validate that all steps have valid tools
        available_tool_names = [tool.name for tool in TOOLS]
        for step in plan.steps:
            if step.tool_name not in available_tool_names:
                # Fix invalid tool names
                step.tool_name = "get_project_info"  # Safe fallback
        
        # USER-FRIENDLY RESPONSE with step preview
        step_preview = f"\n\nHere's my plan:\n"
        for i, step in enumerate(plan.steps):
            step_preview += f"{i+1}. {step.description}\n"
        
        user_response = f"I'll help you with {user_message.lower()}. Let me work through this systematically using my Unity development tools.{step_preview}"
        
        return {
            "plan": plan,
            "step_index": 0,
            "retry_count": 0,
            "messages": [AIMessage(content=user_response)]
        }
    
    except Exception as e:
        # Fallback to analysis-based planning if LLM fails
        fallback_steps = create_tool_aware_plan(user_message, analysis)
        fallback_plan = ExecutionPlan(
            goal=user_message,
            steps=fallback_steps
        )
        
        return {
            "plan": fallback_plan,
            "step_index": 0,
            "retry_count": 0,
            "messages": [AIMessage(content=f"I'll help you with {user_message.lower()}. Let me use my development tools to assist you.")]
        }


async def act(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Execute the current step with both tool calls AND user-facing progress updates."""
    context = runtime.context
    model = get_model(context.model)
    
    if not state.plan or state.step_index >= len(state.plan.steps):
        return {"messages": [AIMessage(content="I'm having trouble processing your request. Could you please try rephrasing it?")]}
    
    current_step = state.plan.steps[state.step_index]
    
    # Ensure the step has a valid tool
    available_tool_names = [tool.name for tool in TOOLS]
    if current_step.tool_name not in available_tool_names:
        current_step.tool_name = "get_project_info"  # Safe fallback
    
    # CREATE USER-FRIENDLY PROGRESS MESSAGE
    step_message = create_user_friendly_step_message(
        current_step, 
        state.step_index + 1, 
        len(state.plan.steps)
    )
    
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
    
    # Prepare messages for the model
    messages = [
        {"role": "system", "content": ACT_PROMPT.format(execution_context=json.dumps(execution_context, indent=2))},
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
            AIMessage(content=step_message),  # User-facing progress update
            response  # Tool call response
        ]
        
        return {
            "plan": updated_plan,
            "messages": messages_to_add,
            "total_tool_calls": state.total_tool_calls + (len(response.tool_calls) if response.tool_calls else 0)
        }
        
    except Exception as e:
        # Handle execution errors
        user_friendly_error = f"**Step {state.step_index + 1}/{len(state.plan.steps)}**: Encountered an issue with this step. Let me try a different approach."
        
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
            "messages": [AIMessage(content=user_friendly_error)],
            "tool_errors": state.tool_errors + [{"step": state.step_index, "error": str(e)}]
        }


# Keep the rest of the functions the same - just the act function needed changes
async def assess(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Assess whether the current step succeeded - INTERNAL ONLY."""
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
    
    # Find the most recent tool messages for better context
    recent_tool_results = []
    for msg in reversed(state.messages):
        if isinstance(msg, ToolMessage):
            recent_tool_results.append(f"Tool '{msg.name}': {get_message_text(msg)}")
            if len(recent_tool_results) >= 3:  # Get last 3 tool results
                break
    
    tool_results_text = "\n".join(recent_tool_results) if recent_tool_results else "No recent tool results found"
    
    # Enhanced assessment focused on step completion
    assessment_request = f"""Assess if the current step has been successfully completed:

Overall Goal: {state.plan.goal}
Current Step ({state.step_index + 1}/{len(state.plan.steps)}): {current_step.description}
Success Criteria: {current_step.success_criteria}

Recent Tool Results:
{tool_results_text}

Last AI Response: {last_result}

Evaluation:
- Was the step's specific objective achieved?
- If a tool was supposed to be used, was it used effectively?
- Is there enough information to move to the next step?
- Did this step contribute meaningfully to the overall goal?

Respond with JSON:
{{
    "outcome": "success|retry|blocked",
    "reason": "specific explanation of assessment",
    "fix": "what needs improvement if retry",
    "confidence": 0.8
}}

Be thorough but fair in assessment."""
    
    messages = [
        {"role": "system", "content": ASSESSMENT_PROMPT},
        {"role": "user", "content": assessment_request}
    ]
    
    try:
        response = await model.ainvoke(messages)
        assessment_text = get_message_text(response)
        
        # Parse assessment
        try:
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
                # Fallback assessment based on keywords and tool usage
                has_tool_results = len(recent_tool_results) > 0
                content_length = len(last_result)
                
                if ("success" in assessment_text.lower() or 
                    (has_tool_results and content_length > 50)):
                    outcome = "success"
                elif "blocked" in assessment_text.lower() or "cannot" in assessment_text.lower():
                    outcome = "blocked"
                else:
                    outcome = "retry"
                
                assessment = AssessmentOutcome(
                    outcome=outcome,
                    reason=assessment_text,
                    confidence=0.6
                )
        except Exception:
            # Default to success if we have tool results, otherwise retry
            has_recent_tools = len(recent_tool_results) > 0
            assessment = AssessmentOutcome(
                outcome="success" if has_recent_tools else "retry",
                reason="Assessment parsing failed, using heuristic",
                confidence=0.4
            )
        
        return {
            "current_assessment": assessment,
            "total_assessments": state.total_assessments + 1
        }
        
    except Exception as e:
        # Fallback assessment - be more optimistic if we have tool activity
        has_tool_activity = any(isinstance(msg, ToolMessage) for msg in state.messages[-5:])
        fallback_assessment = AssessmentOutcome(
            outcome="success" if has_tool_activity else "retry",
            reason=f"Assessment failed: {str(e)}",
            confidence=0.3
        )
        return {
            "current_assessment": fallback_assessment,
            "total_assessments": state.total_assessments + 1
        }


async def repair(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Replan after encountering issues - try alternative approach."""
    context = runtime.context
    model = get_model(context.planning_model or context.model)
    
    if not state.plan:
        return {}
    
    # Create a user-friendly message about trying a different approach
    user_message = "Let me try a different approach to better help you with this."
    
    # Gather information about what went wrong (internal)
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

Respond with a JSON plan like before, focusing on alternative methods."""
    
    messages = [
        {"role": "system", "content": REPAIR_PROMPT},
        {"role": "user", "content": repair_request}
    ]
    
    try:
        response = await model.ainvoke(messages)
        repair_content = get_message_text(response)
        
        # Parse the revised plan
        revised_plan = parse_plan_from_response(repair_content, state.plan.goal)
        
        return {
            "plan": revised_plan,
            "step_index": 0,  # Start from beginning with new plan
            "retry_count": 0,
            "current_assessment": None,
            "plan_revision_count": state.plan_revision_count + 1,
            "messages": [AIMessage(content=user_message)]
        }
        
    except Exception as e:
        # Create a simple fallback plan
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
    """Provide final summary with actual results and insights."""
    context = runtime.context
    model = get_model(context.model)
    
    # Gather information about what was accomplished
    completed_tools = []
    tool_results = []
    for msg in state.messages:
        if isinstance(msg, ToolMessage):
            completed_tools.append(msg.name)
            # Extract key information from tool results
            content = get_message_text(msg)
            if len(content) > 100:  # Only include substantial results
                tool_results.append(f"{msg.name}: {content[:200]}...")
    
    # Check for substantive content in AI messages
    substantive_responses = []
    for msg in state.messages:
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            content = get_message_text(msg)
            if len(content) > 150 and "step" not in content.lower():
                substantive_responses.append(content[:300])
    
    # Create comprehensive summary request
    summary_request = f"""Create a helpful summary of what was accomplished for the user's request: "{state.plan.goal if state.plan else 'Unity development assistance'}"

Tools Used: {', '.join(set(completed_tools)) if completed_tools else 'None'}
Steps Completed: {len(state.completed_steps)} out of {len(state.plan.steps) if state.plan else 0}

Key Results:
{chr(10).join(tool_results[:3]) if tool_results else "No significant tool results captured"}

Provide a summary that:
1. Highlights what was successfully accomplished
2. Mentions any code, assets, or configurations created
3. Provides key insights or recommendations discovered
4. Suggests logical next steps for the user
5. Offers to elaborate on specific aspects

Be specific about Unity/game development outcomes, not just generic task completion."""
    
    messages = [
        {"role": "system", "content": FINAL_SUMMARY_PROMPT},
        {"role": "user", "content": summary_request}
    ]
    
    try:
        response = await model.ainvoke(messages)
        summary_content = get_message_text(response)
        
        return {
            "messages": [AIMessage(content=summary_content)]
        }
        
    except Exception:
        # Fallback summary based on what we know
        if completed_tools:
            fallback_message = f"I've worked through your Unity development request using {', '.join(set(completed_tools))}. The process involved {len(state.completed_steps)} completed steps. Would you like me to elaborate on any specific aspect of what was accomplished, or help you with the next phase of your project?"
        else:
            fallback_message = f"I apologize that I wasn't able to fully complete your Unity development request. Would you like me to try a different approach, or could you help me understand what specific aspect you're most interested in?"
        
        return {
            "messages": [AIMessage(content=fallback_message)]
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


# Routing functions (unchanged)
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


# Graph construction (unchanged)
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