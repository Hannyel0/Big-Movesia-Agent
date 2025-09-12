"""Action nodes for the ReAct agent."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from functools import lru_cache
import hashlib

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
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
from react_agent.utils import get_message_text, get_model


# Initialize narration components
narration_engine = NarrationEngine()

# Conversation summary cache
_conversation_cache: Dict[str, str] = {}


def _generate_message_hash(messages: List) -> str:
    """Generate a hash for message sequence for caching."""
    content = ""
    for msg in messages:
        if isinstance(msg, (HumanMessage, AIMessage, ToolMessage)):
            content += f"{type(msg).__name__}:{get_message_text(msg)[:100]}"
    return hashlib.md5(content.encode()).hexdigest()


def _get_cached_conversation_summary(messages_hash: str, goal: str) -> Optional[str]:
    """Get cached conversation summary if available."""
    cache_key = f"{messages_hash}:{goal}"
    return _conversation_cache.get(cache_key)


def _cache_conversation_summary(messages_hash: str, goal: str, summary: str) -> None:
    """Cache conversation summary."""
    cache_key = f"{messages_hash}:{goal}"
    _conversation_cache[cache_key] = summary
    # Keep cache size reasonable
    if len(_conversation_cache) > 100:
        # Remove oldest entries (simple FIFO)
        keys_to_remove = list(_conversation_cache.keys())[:20]
        for key in keys_to_remove:
            del _conversation_cache[key]


def _build_conversation_context(
    state: State, current_step: PlanStep
) -> List[Dict[str, str]]:
    """Build optimized conversation context with caching and limited message history."""
    context_messages = []

    # Only include essential goal context - just the first user message
    original_request = None
    for msg in state.messages:
        if isinstance(msg, HumanMessage):
            original_request = get_message_text(msg)
            break

    # Limit to only 2-3 most recent relevant messages instead of 6
    recent_messages = state.messages[-3:] if len(state.messages) > 3 else state.messages

    # Check if we have a cached summary for older conversation
    if len(state.messages) > 5:
        older_messages = state.messages[1:-3]  # Exclude first and last 3
        messages_hash = _generate_message_hash(older_messages)
        cached_summary = _get_cached_conversation_summary(
            messages_hash, state.plan.goal
        )

        if cached_summary:
            # Use cached summary instead of processing all messages
            context_messages.append(
                {
                    "role": "system",
                    "content": f"Previous conversation summary: {cached_summary}",
                }
            )
        else:
            # Generate and cache summary for older messages
            summary = _generate_conversation_summary(older_messages)
            _cache_conversation_summary(messages_hash, state.plan.goal, summary)
            if summary:
                context_messages.append(
                    {
                        "role": "system",
                        "content": f"Previous conversation summary: {summary}",
                    }
                )

    # Add original request for goal context (shortened)
    if original_request:
        # Truncate if too long to save tokens
        truncated_request = (
            original_request[:200] + "..."
            if len(original_request) > 200
            else original_request
        )
        context_messages.append({"role": "user", "content": truncated_request})

    # Add only the most recent 2-3 relevant messages with minimal detail
    relevant_count = 0
    for msg in reversed(recent_messages):
        if relevant_count >= 2:  # Limit to 2 most relevant recent messages
            break

        if isinstance(msg, AIMessage):
            content = get_message_text(msg)
            if content and not msg.tool_calls and len(content.strip()) > 20:
                # Truncate long messages to save tokens
                truncated_content = (
                    content[:150] + "..." if len(content) > 150 else content
                )
                context_messages.append(
                    {"role": "assistant", "content": truncated_content}
                )
                relevant_count += 1
            elif msg.tool_calls:
                # Very brief tool call summary
                tool_names = [
                    tc.get("name", "unknown") for tc in msg.tool_calls[:2]
                ]  # Limit to first 2 tools
                context_messages.append(
                    {
                        "role": "assistant",
                        "content": f"[Used tools: {', '.join(tool_names)}]",
                    }
                )
                relevant_count += 1
        elif isinstance(msg, ToolMessage) and relevant_count < 2:
            # Very brief tool result
            tool_name = msg.name or "tool"
            try:
                result = json.loads(get_message_text(msg))
                success = result.get("success", True)
                status = "succeeded" if success else "failed"
                context_messages.append(
                    {"role": "user", "content": f"[{tool_name} {status}]"}
                )
            except:
                context_messages.append(
                    {"role": "user", "content": f"[{tool_name} completed]"}
                )
            relevant_count += 1

    return context_messages


def _generate_conversation_summary(messages: List) -> str:
    """Generate a brief summary of conversation messages for caching."""
    if not messages:
        return ""

    # Count different types of activities
    tool_calls = 0
    user_inputs = 0
    ai_responses = 0

    key_topics = set()

    for msg in messages:
        if isinstance(msg, HumanMessage):
            user_inputs += 1
            # Extract key terms for context
            content = get_message_text(msg).lower()
            for term in [
                "unity",
                "unreal",
                "script",
                "shader",
                "game",
                "asset",
                "scene",
            ]:
                if term in content:
                    key_topics.add(term)
        elif isinstance(msg, AIMessage):
            ai_responses += 1
            if msg.tool_calls:
                tool_calls += len(msg.tool_calls)
        elif isinstance(msg, ToolMessage):
            # Already counted in tool_calls
            pass

    # Generate concise summary
    summary_parts = []
    if tool_calls > 0:
        summary_parts.append(f"{tool_calls} tool operations")
    if user_inputs > 1:
        summary_parts.append(f"{user_inputs} user inputs")
    if key_topics:
        summary_parts.append(f"Topics: {', '.join(sorted(key_topics))}")

    return "; ".join(summary_parts) if summary_parts else "Previous conversation steps"


def _create_context_aware_narration_prompt(
    current_step: PlanStep,
    step_context: Dict[str, Any],
    retry_count: int,
    assessment: Any,
) -> str:
    """Create a context-aware prompt for narration generation."""
    base_prompt = f"""You are providing live development commentary for step {step_context["step_index"] + 1} of {step_context["total_steps"]}.

Current step: {current_step.description}
Tool to use: {current_step.tool_name}
Goal: {step_context["goal"]}"""

    # Add retry context if applicable
    if retry_count > 0:
        if assessment and hasattr(assessment, "reason"):
            base_prompt += f"\n\nThis is retry attempt #{retry_count + 1}. Previous attempt had issues: {assessment.reason}"
        else:
            base_prompt += f"\n\nThis is retry attempt #{retry_count + 1}. Trying again with a different approach."

    base_prompt += """\n\nGenerate a brief, engaging commentary (1-2 sentences) that:
- Acknowledges the conversation context and what has happened so far
- Explains what you're about to do with the specific tool
- Shows awareness of any previous attempts or results
- Sounds knowledgeable about game development

Be natural and conversational while showing you understand the current situation."""

    return base_prompt


async def act_with_narration_guard(
    state: State, runtime: Runtime[Context]
) -> Dict[str, Any]:
    """Enhanced act node with context-aware narration alongside tool calls."""
    context = runtime.context
    model = get_model(context.model)

    if not state.plan or state.step_index >= len(state.plan.steps):
        return {
            "messages": [
                AIMessage(
                    content="I'm having trouble processing your request. Could you please try rephrasing it?"
                )
            ]
        }

    current_step = state.plan.steps[state.step_index]

    # Create step context for narration
    step_context = {
        "step_index": state.step_index,
        "total_steps": len(state.plan.steps),
        "goal": state.plan.goal,
        "tool_name": current_step.tool_name,
        "description": current_step.description,
    }

    # STRATEGY: Two-phase approach with conversation context

    # Phase 1: Generate contextual narration WITH conversation history
    try:
        # Build conversation context for narration
        conversation_context = _build_conversation_context(state, current_step)

        # Create context-aware narration prompt
        narration_prompt = _create_context_aware_narration_prompt(
            current_step, step_context, state.retry_count, state.current_assessment
        )

        # Include conversation context in narration generation
        narration_messages = [
            {
                "role": "system",
                "content": "You are providing live development commentary. Generate a brief (1-2 sentences), engaging message about what you're about to do. Be specific about the Unity/Unreal tool you'll use and show awareness of the conversation context.",
            },
            *conversation_context,  # Include conversation history!
            {"role": "user", "content": narration_prompt},
        ]

        narration_response = await model.ainvoke(narration_messages)
        llm_generated_narration = get_message_text(narration_response)

        # Validate the LLM narration quality
        if len(llm_generated_narration.strip()) < 15 or _is_generic_response(
            llm_generated_narration
        ):
            # Fallback to engine-generated narration if LLM narration is poor
            llm_generated_narration = _create_rich_pre_step_narration(
                current_step, step_context, state.retry_count
            )

    except Exception as e:
        # Fallback to pre-generated narration if LLM narration fails
        llm_generated_narration = _create_rich_pre_step_narration(
            current_step, step_context, state.retry_count
        )

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

    # Prepare minimal conversation context for tool usage (only 2 most recent)
    conversation_messages = []
    relevant_messages = []

    # Get only the 2 most recent relevant messages to minimize tokens
    for msg in reversed(state.messages[-3:]):
        if len(relevant_messages) >= 2:
            break
        if isinstance(msg, HumanMessage):
            content = get_message_text(msg)
            # Truncate long messages
            truncated = content[:100] + "..." if len(content) > 100 else content
            relevant_messages.append({"role": "user", "content": truncated})
        elif isinstance(msg, AIMessage) and not msg.tool_calls:
            content = get_message_text(msg)
            if len(content.strip()) > 20:  # Only include substantial responses
                truncated = content[:100] + "..." if len(content) > 100 else content
                relevant_messages.append({"role": "assistant", "content": truncated})

    # Reverse to maintain chronological order
    conversation_messages = list(reversed(relevant_messages))

    # Dynamic context for tool execution
    execution_context = {
        "current_step": current_step.description,
        "success_criteria": current_step.success_criteria,
        "step_number": state.step_index + 1,
        "total_steps": len(state.plan.steps),
        "required_tool": current_step.tool_name,
        "available_tools": [tool.name for tool in TOOLS],
        "retry_count": state.retry_count,
    }

    dynamic_context_message = f"""Current execution context:
{json.dumps(execution_context, indent=2)}"""

    # Messages for tool execution
    tool_messages = [
        {"role": "system", "content": static_system_content},
        {"role": "system", "content": dynamic_context_message},
        *conversation_messages,
        {"role": "user", "content": tool_call_prompt},
    ]

    try:
        # Bind tools and force the specific tool usage
        model_with_tools = model.bind_tools(TOOLS)
        tool_response = await model_with_tools.ainvoke(
            tool_messages,
            tool_choice={
                "type": "function",
                "function": {"name": current_step.tool_name},
            },
        )

        # Combine narration with tool call
        combined_response = AIMessage(
            content=llm_generated_narration,
            tool_calls=tool_response.tool_calls,
            additional_kwargs=tool_response.additional_kwargs,
            response_metadata=tool_response.response_metadata,
            id=tool_response.id,
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
            error_messages=current_step.error_messages,
        )

        updated_plan = ExecutionPlan(
            goal=state.plan.goal, steps=updated_steps, metadata=state.plan.metadata
        )

        return {
            "plan": updated_plan,
            "messages": [combined_response],
            "total_tool_calls": state.total_tool_calls
            + (len(tool_response.tool_calls) if tool_response.tool_calls else 0),
        }

    except Exception as e:
        # Rich error narration with context
        retry_text = (
            f" (retry #{state.retry_count + 1})" if state.retry_count > 0 else ""
        )
        error_narration = f"Hit a snag while {current_step.description.lower()}{retry_text}: {str(e)}. Let me try a different approach."

        updated_steps = list(state.plan.steps)
        error_messages = current_step.error_messages + [str(e)]
        updated_steps[state.step_index] = PlanStep(
            description=current_step.description,
            tool_name=current_step.tool_name,
            success_criteria=current_step.success_criteria,
            dependencies=current_step.dependencies,
            status=current_step.status,
            attempts=current_step.attempts + 1,
            error_messages=error_messages,
        )

        updated_plan = ExecutionPlan(
            goal=state.plan.goal, steps=updated_steps, metadata=state.plan.metadata
        )

        return {
            "plan": updated_plan,
            "messages": [AIMessage(content=error_narration)],
            "tool_errors": state.tool_errors
            + [{"step": state.step_index, "error": str(e)}],
        }


async def act(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Act node with context-aware narration guard approach."""
    return await act_with_narration_guard(state, runtime)


def _create_tool_execution_prompt(
    current_step: PlanStep, step_context: Dict[str, Any]
) -> str:
    """Create a focused prompt for tool execution without narration distractions."""
    return f"""Execute step {step_context["step_index"] + 1}: {current_step.description}

Required Tool: {current_step.tool_name}
Success Criteria: {current_step.success_criteria}

Call the {current_step.tool_name} tool with appropriate parameters to complete this development step. Focus on creating working game development deliverables that meet the success criteria."""


def _is_generic_response(narration: str) -> bool:
    """Check if the generated narration is too generic or low-quality."""
    generic_phrases = [
        "i'll help you",
        "let me assist",
        "i can help",
        "sure, i can",
        "i'll do that",
        "let me do that",
        "i'll work on",
        "let me work on",
        "i'll try to",
        "let me try",
        "i'll attempt",
        "let me attempt",
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
        "unity",
        "unreal",
        "script",
        "shader",
        "material",
        "prefab",
        "scene",
        "gameobject",
        "component",
        "asset",
        "texture",
        "mesh",
        "animation",
        "physics",
        "collider",
        "rigidbody",
        "transform",
        "canvas",
        "ui",
    ]

    has_game_dev_context = any(term in narration_lower for term in game_dev_terms)

    return not has_game_dev_context


def _create_rich_pre_step_narration(
    current_step: PlanStep, step_context: Dict[str, Any], retry_count: int = 0
) -> str:
    """Create rich, contextual narration for a development step with retry awareness."""
    # Tool-specific narration templates
    base_narrations = {
        "search": f"Searching for Unity/Unreal resources and tutorials related to: {current_step.description}",
        "get_project_info": f"Analyzing the current project structure and configuration to understand: {current_step.description}",
        "get_script_snippets": f"Retrieving code templates and examples for: {current_step.description}",
        "create_asset": f"Creating a new game asset: {current_step.description}",
        "write_file": f"Writing code for: {current_step.description}",
        "scene_management": f"Setting up scene elements for: {current_step.description}",
        "compile_and_test": f"Compiling and testing the implementation: {current_step.description}",
        "edit_project_config": f"Configuring project settings for: {current_step.description}",
    }

    base_narration = base_narrations.get(
        current_step.tool_name,
        f"Executing {current_step.tool_name} for: {current_step.description}",
    )

    # Add retry context if applicable
    if retry_count > 0:
        retry_prefixes = {
            1: "Trying a different approach - ",
            2: "Let me retry this with adjusted parameters - ",
            3: "Making another attempt - ",
        }
        prefix = retry_prefixes.get(retry_count, f"Attempt #{retry_count + 1} - ")
        base_narration = prefix + base_narration.lower()

    # Add step context
    step_info = (
        f" (Step {step_context['step_index'] + 1} of {step_context['total_steps']})"
    )

    return base_narration + step_info
