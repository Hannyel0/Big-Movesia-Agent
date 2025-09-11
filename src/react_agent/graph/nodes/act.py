"""Action nodes for the ReAct agent."""

from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.state import (
    ExecutionPlan,
    PlanStep,
    State,
    StepStatus,
)
from react_agent.tools import TOOLS
from react_agent.narration import NarrationEngine
from react_agent.utils import get_message_text, get_model, _create_rich_pre_step_narration


# Initialize narration components
narration_engine = NarrationEngine()


async def act_with_narration_guard(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Enhanced act node with guardrails to ensure LLM-generated narration alongside tool calls."""
    context = runtime.context
    model = get_model(context.model)
    
    if not state.plan or state.step_index >= len(state.plan.steps):
        return {"messages": [AIMessage(content="I'm having trouble processing your request. Could you please try rephrasing it?")]}
    
    current_step = state.plan.steps[state.step_index]
    
    # Create step context for narration
    step_context = {
        "step_index": state.step_index,
        "total_steps": len(state.plan.steps),
        "goal": state.plan.goal,
        "tool_name": current_step.tool_name,
        "description": current_step.description
    }
    
    # STRATEGY: Two-phase approach for guaranteed narration + tool call
    
    # Phase 1: Generate contextual narration
    narration_prompt = _create_narration_prompt(current_step, step_context)
    
    try:
        # Get LLM-generated narration first (without tools to avoid distraction)
        narration_messages = [
            {"role": "system", "content": "You are providing live development commentary. Generate a brief (1-2 sentences), engaging message about what you're about to do. Be specific about the Unity/Unreal tool you'll use and sound knowledgeable."},
            {"role": "user", "content": narration_prompt}
        ]
        
        narration_response = await model.ainvoke(narration_messages)
        llm_generated_narration = get_message_text(narration_response)
        
        # Validate the LLM narration quality
        if len(llm_generated_narration.strip()) < 15 or _is_generic_response(llm_generated_narration):
            # Fallback to engine-generated narration if LLM narration is poor
            llm_generated_narration = _create_rich_pre_step_narration(current_step, step_context)
        
    except Exception as e:
        # Fallback to pre-generated narration if LLM narration fails
        llm_generated_narration = _create_rich_pre_step_narration(current_step, step_context)
    
    # Phase 2: Generate tool call with execution context
    tool_call_prompt = _create_tool_execution_prompt(current_step, step_context)
    
    # Static system content for tool execution - cacheable
    static_system_content = """You are executing a Unity/Unreal Engine development step. Focus on calling the required tool with appropriate parameters.

EXECUTION GUIDELINES:
1. Call the exact tool specified in the step requirements  
2. Use appropriate parameters based on the step description and success criteria
3. Focus on creating working game development deliverables
4. Follow Unity/Unreal best practices in your tool usage

You MUST call the specified tool to complete this development step."""

    # Prepare conversation context for better tool usage
    conversation_messages = []
    for msg in state.messages[-3:]:
        if isinstance(msg, HumanMessage):
            conversation_messages.append({"role": "user", "content": get_message_text(msg)})
        elif isinstance(msg, AIMessage) and not msg.tool_calls:
            conversation_messages.append({"role": "assistant", "content": get_message_text(msg)})
    
    # Dynamic context for tool execution
    execution_context = {
        "current_step": current_step.description,
        "success_criteria": current_step.success_criteria,
        "step_number": state.step_index + 1,
        "total_steps": len(state.plan.steps),
        "required_tool": current_step.tool_name,
        "available_tools": [tool.name for tool in TOOLS]
    }
    
    dynamic_context_message = f"""Current execution context:
{json.dumps(execution_context, indent=2)}"""
    
    # Messages for tool execution
    tool_messages = [
        {"role": "system", "content": static_system_content},
        {"role": "system", "content": dynamic_context_message},
        *conversation_messages,
        {"role": "user", "content": tool_call_prompt}
    ]
    
    try:
        # Bind tools and force the specific tool usage
        model_with_tools = model.bind_tools(TOOLS)
        tool_response = await model_with_tools.ainvoke(
            tool_messages,
            tool_choice={"type": "function", "function": {"name": current_step.tool_name}}
        )
        
        # Combine narration with tool call
        combined_response = AIMessage(
            content=llm_generated_narration,
            tool_calls=tool_response.tool_calls,
            additional_kwargs=tool_response.additional_kwargs,
            response_metadata=tool_response.response_metadata,
            id=tool_response.id
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
        
        return {
            "plan": updated_plan,
            "messages": [combined_response],
            "total_tool_calls": state.total_tool_calls + (len(tool_response.tool_calls) if tool_response.tool_calls else 0)
        }
        
    except Exception as e:
        # Rich error narration
        error_narration = f"Hit a snag while {current_step.description.lower()}: {str(e)}. Let me try a different approach."
        
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


async def act(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Act node with two-phase narration guard approach for guaranteed LLM narration and tool calls."""
    return await act_with_narration_guard(state, runtime)


def _create_narration_prompt(current_step: PlanStep, step_context: Dict[str, Any]) -> str:
    """Create a prompt for generating contextual narration about the current step."""
    return f"""You are about to execute step {step_context['step_index'] + 1} of {step_context['total_steps']} for the goal: "{step_context['goal']}"

Step Description: {current_step.description}
Tool to Use: {current_step.tool_name}
Success Criteria: {current_step.success_criteria}

Generate a brief, engaging commentary (1-2 sentences) about what you're about to do. Be specific about the Unity/Unreal tool you'll use and sound knowledgeable about game development.

Example styles:
- "I'll search for the latest Unity water shader techniques to find advanced approaches for realistic water rendering."
- "Let me create a new C# script for the player movement controller with smooth input handling and physics-based movement."
- "I'll analyze the current project structure to understand the existing architecture and identify the best location for the new gameplay systems."

Your commentary:"""


def _create_tool_execution_prompt(current_step: PlanStep, step_context: Dict[str, Any]) -> str:
    """Create a focused prompt for tool execution without narration distractions."""
    return f"""Execute step {step_context['step_index'] + 1}: {current_step.description}

Required Tool: {current_step.tool_name}
Success Criteria: {current_step.success_criteria}

Call the {current_step.tool_name} tool with appropriate parameters to complete this development step. Focus on creating working game development deliverables that meet the success criteria."""


def _is_generic_response(narration: str) -> bool:
    """Check if the generated narration is too generic or low-quality."""
    generic_phrases = [
        "i'll help you", "let me assist", "i can help", "sure, i can",
        "i'll do that", "let me do that", "i'll work on", "let me work on",
        "i'll try to", "let me try", "i'll attempt", "let me attempt",
        "i'll start by", "let me start", "i'll begin", "let me begin"
    ]

    narration_lower = narration.lower().strip()

    # Check for generic phrases
    for phrase in generic_phrases:
        if phrase in narration_lower:
            return True

    # Check if it's too short or lacks specificity
    if len(narration_lower) < 20:
        return True

    # Check if it mentions the specific tool or Unity/Unreal concepts
    game_dev_terms = [
        "unity", "unreal", "script", "shader", "material", "prefab", "scene",
        "gameobject", "component", "asset", "texture", "mesh", "animation",
        "physics", "collider", "rigidbody", "transform", "canvas", "ui"
    ]

    has_game_dev_context = any(term in narration_lower for term in game_dev_terms)

    return not has_game_dev_context


def _create_rich_pre_step_narration(current_step: PlanStep, step_context: Dict[str, Any]) -> str:
    """Create rich, contextual narration for a development step using the narration engine."""
    # Tool-specific narration templates
    tool_narrations = {
        "search": f"Searching for Unity/Unreal resources and tutorials related to: {current_step.description}",
        "get_project_info": f"Analyzing the current project structure and configuration to understand: {current_step.description}",
        "get_script_snippets": f"Retrieving code templates and examples for: {current_step.description}",
        "create_asset": f"Creating a new game asset: {current_step.description}",
        "write_file": f"Writing code for: {current_step.description}",
        "scene_management": f"Setting up scene elements for: {current_step.description}",
        "compile_and_test": f"Compiling and testing the implementation: {current_step.description}",
        "edit_project_config": f"Configuring project settings for: {current_step.description}"
    }

    base_narration = tool_narrations.get(
        current_step.tool_name,
        f"Executing {current_step.tool_name} for: {current_step.description}"
    )

    # Add step context
    step_info = f" (Step {step_context['step_index'] + 1} of {step_context['total_steps']})"

    return base_narration + step_info