"""Enhanced act.py with Tier 0 micro-retry integration."""

import json
import logging
import asyncio
from typing import Any, Dict, List, Optional
from functools import lru_cache
import hashlib

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.state import ExecutionPlan, PlanStep, State, StepStatus
from react_agent.tools import TOOLS
from react_agent.narration import NarrationEngine
from react_agent.utils import get_message_text, get_model, is_anthropic_model
from react_agent.memory import inject_memory_into_prompt
from react_agent.ui_messages import create_plan_ui_message


# Initialize narration components
narration_engine = NarrationEngine()
logger = logging.getLogger(__name__)

# TIER 0: Micro-retry configuration
MICRO_RETRY_CONFIG = {
    "max_attempts": 2,
    "backoff_base": 1.0,  # seconds
    "transient_error_patterns": [
        "network",
        "timeout",
        "connection",
        "rate limit",
        "service unavailable",
        "temporary",
        "try again",
    ],
    "retryable_categories": ["network_error", "tool_malfunction", "timeout"],
}


def _is_transient_error(tool_result: Dict[str, Any], tool_name: str) -> bool:
    """Detect if this is a transient error that should get micro-retry."""
    if tool_result.get("success", True):
        return False

    error_message = tool_result.get("error", "").lower()
    error_category = tool_result.get("error_category", "").lower()

    # Check error category first
    if error_category in MICRO_RETRY_CONFIG["retryable_categories"]:
        return True

    # Check error message patterns
    for pattern in MICRO_RETRY_CONFIG["transient_error_patterns"]:
        if pattern in error_message:
            return True

    return False


async def _micro_retry_tool_call(
    tool_response: AIMessage,
    runtime: Runtime[Context],
    current_step: PlanStep,
    attempt_num: int = 1,
) -> AIMessage:
    """Execute micro-retry for transient tool failures (Tier 0)."""
    model = get_model(runtime.context.model)

    if attempt_num > MICRO_RETRY_CONFIG["max_attempts"]:
        return tool_response  # Give up, let normal error handling take over

    # Exponential backoff
    if attempt_num > 1:
        delay = MICRO_RETRY_CONFIG["backoff_base"] * (2 ** (attempt_num - 2))
        await asyncio.sleep(delay)

    try:
        # Retry with same tool call but slight parameter adjustment for robustness
        model_with_tools = model.bind_tools(TOOLS)

        # Create slightly adjusted prompt for retry
        retry_prompt = f"""Retry executing: {current_step.description}

This is retry attempt #{attempt_num} due to transient error.
Use the {current_step.tool_name} tool with robust parameters.
Focus on completing: {current_step.success_criteria}"""

        # ✅ CACHING: Cache the retry system prompt for Anthropic models only
        cache_enabled = getattr(runtime.context, "enable_prompt_cache", True)
        using_anthropic = is_anthropic_model(runtime.context.model)
        retry_system_content = "You are retrying a tool call due to transient error. Use the same tool with slightly adjusted parameters for robustness."

        if cache_enabled and using_anthropic:
            retry_system_structured = [
                {
                    "type": "text",
                    "text": retry_system_content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            retry_system_structured = retry_system_content

        retry_response = await model_with_tools.ainvoke(
            [
                {"role": "system", "content": retry_system_structured},
                {"role": "user", "content": retry_prompt},
            ],
            tool_choice={
                "type": "function",
                "function": {"name": current_step.tool_name},
            },
        )

        return retry_response

    except Exception as e:
        # If retry also fails, return original response
        return tool_response


# Conversation summary cache (keeping existing functionality)
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
    if len(_conversation_cache) > 100:
        keys_to_remove = list(_conversation_cache.keys())[:20]
        for key in keys_to_remove:
            del _conversation_cache[key]


def _detect_continuation_request(
    user_message: str, recent_messages: List
) -> Optional[Dict[str, Any]]:
    """Detect if user is responding to a previous offer/question.

    Returns context about what they're continuing if detected.
    """
    message_lower = user_message.lower().strip()

    # Short affirmative responses
    affirmatives = [
        "yes",
        "yeah",
        "yep",
        "sure",
        "ok",
        "okay",
        "please",
        "yes please",
        "yeah please",
        "show me",
        "go ahead",
        "do it",
        "let's see it",
        "i want to see",
        "i'd like to see",
    ]

    # Check if this is a continuation
    is_continuation = any(message_lower.startswith(aff) for aff in affirmatives) or any(
        aff in message_lower for aff in ["yes show", "yeah show", "please show"]
    )

    if not is_continuation:
        return None

    # Look for what was offered in previous AI message
    for msg in reversed(recent_messages[-5:]):
        if isinstance(msg, AIMessage):
            content = get_message_text(msg)
            content_lower = content.lower()

            # Check for offers/questions
            if any(
                phrase in content_lower
                for phrase in [
                    "want me to",
                    "would you like",
                    "should i show",
                    "want to see",
                    "i can show",
                    "let me know if",
                ]
            ):
                # Found the offer - now find the associated tool result
                return {
                    "type": "continuation",
                    "previous_offer": content,
                    "context_needed": True,
                }

    return None


def _extract_recent_tool_results(
    messages: List, limit: int = 2
) -> List[Dict[str, Any]]:
    """Extract recent tool results with full data, not just status."""
    tool_results = []

    for msg in reversed(messages[-10:]):  # Look back further
        if len(tool_results) >= limit:
            break

        if isinstance(msg, ToolMessage):
            try:
                result = json.loads(get_message_text(msg))
                tool_name = msg.name or "unknown"

                # Store full result, not just success status
                tool_results.append(
                    {
                        "tool_name": tool_name,
                        "result": result,
                        "success": result.get("success", True),
                    }
                )
            except:
                continue

    return list(reversed(tool_results))


def _build_conversation_context(
    state: State, current_step: PlanStep, user_message: str = None
) -> List[Dict[str, str]]:
    """Build enhanced conversation context with continuation detection."""
    context_messages = []

    # Check if this is a continuation request
    continuation = None
    if user_message:
        continuation = _detect_continuation_request(user_message, state.messages)

    # If continuation, include MORE context
    if continuation:
        # Include last 5 messages instead of 3
        recent_messages = (
            state.messages[-5:] if len(state.messages) > 5 else state.messages
        )

        # Get full tool results from recent exchanges
        recent_tool_results = _extract_recent_tool_results(state.messages, limit=3)

        # Add tool results to context with FULL DATA
        if recent_tool_results:
            for tool_result in recent_tool_results:
                tool_name = tool_result["tool_name"]
                result = tool_result["result"]

                # Create rich context summary based on tool type
                if tool_name == "code_snippets":
                    snippets = result.get("snippets", [])
                    if snippets:
                        # Include actual snippet data
                        context_messages.append(
                            {
                                "role": "system",
                                "content": f"Recent tool result: code_snippets found {len(snippets)} results. "
                                f"Top result: {snippets[0].get('file_path', 'unknown')} "
                                f"with {len(snippets[0].get('code', ''))} chars of code available.",
                            }
                        )

                elif tool_name == "search_project":
                    results = result.get("results", [])
                    if results:
                        context_messages.append(
                            {
                                "role": "system",
                                "content": f"Recent tool result: search_project found {len(results)} items. "
                                f"Results include: {', '.join([r.get('name', 'unknown') for r in results[:3]])}",
                            }
                        )

                elif tool_name in [
                    "read_file",
                    "write_file",
                    "modify_file",
                    "delete_file",
                    "move_file",
                ]:
                    context_messages.append(
                        {
                            "role": "system",
                            "content": f"Recent tool result: {tool_name} "
                            f"on {result.get('file_path', 'file')}",
                        }
                    )

    else:
        # Normal context (existing logic, slightly enhanced)
        recent_messages = (
            state.messages[-3:] if len(state.messages) > 3 else state.messages
        )

        # Check for cached summary for older conversation
        if len(state.messages) > 5:
            older_messages = state.messages[1:-3]
            messages_hash = _generate_message_hash(older_messages)
            cached_summary = _get_cached_conversation_summary(
                messages_hash, state.plan.goal
            )

            if cached_summary:
                context_messages.append(
                    {
                        "role": "system",
                        "content": f"Previous conversation summary: {cached_summary}",
                    }
                )
            else:
                summary = _generate_conversation_summary(older_messages)
                _cache_conversation_summary(messages_hash, state.plan.goal, summary)
                if summary:
                    context_messages.append(
                        {
                            "role": "system",
                            "content": f"Previous conversation summary: {summary}",
                        }
                    )

    # Add plan goal for context (existing)
    if state.plan and state.plan.goal:
        truncated_goal = (
            state.plan.goal[:200] + "..."
            if len(state.plan.goal) > 200
            else state.plan.goal
        )
        context_messages.append({"role": "user", "content": truncated_goal})

    # Add recent conversation with LESS aggressive truncation for continuations
    relevant_count = 0
    max_relevant = 3 if not continuation else 5  # More context for continuations

    for msg in reversed(recent_messages):
        if relevant_count >= max_relevant:
            break

        if isinstance(msg, AIMessage):
            content = get_message_text(msg)
            if content and not msg.tool_calls and len(content.strip()) > 20:
                # For continuations, keep more of the AI's response
                max_length = 300 if continuation else 150
                truncated_content = (
                    content[:max_length] + "..."
                    if len(content) > max_length
                    else content
                )
                context_messages.append(
                    {"role": "assistant", "content": truncated_content}
                )
                relevant_count += 1
            elif msg.tool_calls:
                tool_names = [tc.get("name", "unknown") for tc in msg.tool_calls[:2]]
                context_messages.append(
                    {
                        "role": "assistant",
                        "content": f"[Used tools: {', '.join(tool_names)}]",
                    }
                )
                relevant_count += 1

        elif isinstance(msg, HumanMessage):
            content = get_message_text(msg)
            # Keep full user messages for continuations
            max_length = 200 if continuation else 100
            truncated = (
                content[:max_length] + "..." if len(content) > max_length else content
            )
            context_messages.append({"role": "user", "content": truncated})
            relevant_count += 1

        elif isinstance(msg, ToolMessage) and relevant_count < max_relevant:
            tool_name = msg.name or "tool"

            # For continuations, include tool result summaries
            if continuation:
                try:
                    result = json.loads(get_message_text(msg))
                    success = result.get("success", True)

                    if tool_name == "code_snippets" and success:
                        total = result.get("total_found", 0)
                        context_messages.append(
                            {
                                "role": "user",
                                "content": f"[code_snippets: found {total} scripts]",
                            }
                        )
                    elif tool_name == "search_project" and success:
                        count = result.get("result_count", 0)
                        context_messages.append(
                            {
                                "role": "user",
                                "content": f"[search_project: found {count} items]",
                            }
                        )
                    else:
                        status = "succeeded" if success else "failed"
                        context_messages.append(
                            {"role": "user", "content": f"[{tool_name} {status}]"}
                        )
                except:
                    context_messages.append(
                        {"role": "user", "content": f"[{tool_name} completed]"}
                    )
            else:
                # Existing minimal tool result handling
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

    tool_calls = 0
    user_inputs = 0
    ai_responses = 0
    key_topics = set()

    for msg in messages:
        if isinstance(msg, HumanMessage):
            user_inputs += 1
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
    """Enhanced act node with micro-retry capability and context-aware narration."""
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

    # CRITICAL: Mark step as IN_PROGRESS immediately before execution
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

    in_progress_plan = ExecutionPlan(
        goal=state.plan.goal, steps=updated_steps, metadata=state.plan.metadata
    )

    # Emit IN_PROGRESS plan UI message BEFORE tool execution (reuse existing ID)
    message_id = state.messages[-1].id if state.messages else ""

    in_progress_ui_msg = create_plan_ui_message(
        in_progress_plan, message_id, ui_message_id=state.plan_ui_message_id
    )

    # Create step context for narration
    step_context = {
        "step_index": state.step_index,
        "total_steps": len(state.plan.steps),
        "goal": state.plan.goal,
        "tool_name": current_step.tool_name,
        "description": current_step.description,
    }

    # Phase 1: Generate contextual narration WITH conversation history
    # Extract user message for continuation detection
    user_message = None
    for msg in reversed(state.messages[-3:]):
        if isinstance(msg, HumanMessage):
            user_message = get_message_text(msg)
            break

    try:
        conversation_context = _build_conversation_context(
            state, current_step, user_message
        )
        narration_prompt = _create_context_aware_narration_prompt(
            current_step, step_context, state.retry_count, state.current_assessment
        )

        # ✅ CACHING: Cache narration system prompt for Anthropic models only
        cache_enabled = getattr(runtime.context, "enable_prompt_cache", True)
        using_anthropic = is_anthropic_model(runtime.context.model)
        narration_system_content = "You are providing live development commentary. Generate a brief (1-2 sentences), engaging message about what you're about to do. Be specific about the Unity/Unreal tool you'll use and show awareness of the conversation context."

        if cache_enabled and using_anthropic:
            narration_system_structured = [
                {
                    "type": "text",
                    "text": narration_system_content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            narration_system_structured = narration_system_content

        narration_messages = [
            {
                "role": "system",
                "content": narration_system_structured,
            },
            *conversation_context,
            {"role": "user", "content": narration_prompt},
        ]

        narration_response = await model.ainvoke(narration_messages)
        llm_generated_narration = get_message_text(narration_response)

        if len(llm_generated_narration.strip()) < 15 or _is_generic_response(
            llm_generated_narration
        ):
            llm_generated_narration = _create_rich_pre_step_narration(
                current_step, step_context, state.retry_count
            )

    except Exception as e:
        llm_generated_narration = _create_rich_pre_step_narration(
            current_step, step_context, state.retry_count
        )

    # Phase 2: Generate tool call with execution context
    tool_call_prompt = _create_tool_execution_prompt(current_step, step_context)

    static_system_content = """## Tool Execution Mode

You are executing a Unity/Unreal Engine development step. Focus on calling the required tool with appropriate parameters.

### EXECUTION GUIDELINES:

1. **Call the exact tool** specified in the step requirements
2. **Use appropriate parameters** based on the step description and success criteria
3. **Focus on creating working** game development deliverables
4. **Follow Unity/Unreal best practices** in your tool usage

⚠️ **CRITICAL**: You MUST call the specified tool to complete this development step.

### MARKDOWN FORMATTING REQUIRED WHEN GENERATING NARRATION OR EXPLANATION"""

    # Prepare minimal conversation context for tool usage
    conversation_messages = []
    relevant_messages = []

    for msg in reversed(state.messages[-3:]):
        if len(relevant_messages) >= 2:
            break
        if isinstance(msg, HumanMessage):
            content = get_message_text(msg)
            truncated = content[:100] + "..." if len(content) > 100 else content
            relevant_messages.append({"role": "user", "content": truncated})
        elif isinstance(msg, AIMessage) and not msg.tool_calls:
            content = get_message_text(msg)
            if len(content.strip()) > 20:
                truncated = content[:100] + "..." if len(content) > 100 else content
                relevant_messages.append({"role": "assistant", "content": truncated})

    conversation_messages = list(reversed(relevant_messages))

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

    # ✅ OPTIMIZATION: Enhance system content with memory context
    enhanced_system_content = await inject_memory_into_prompt(
        base_prompt=static_system_content,
        state=state,
        include_patterns=True,
        include_episodes=False,  # Keep it light for tool execution
    )

    # ✅ CACHING: Check if caching is enabled AND using Anthropic
    cache_enabled = getattr(runtime.context, "enable_prompt_cache", True)
    using_anthropic = is_anthropic_model(runtime.context.model)

    # Create cacheable system content for the static part (Anthropic only)
    if cache_enabled and using_anthropic:
        # Use structured content format with cache control for the static system prompt
        enhanced_system_structured = [
            {
                "type": "text",
                "text": enhanced_system_content,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    else:
        enhanced_system_structured = enhanced_system_content

    tool_messages = [
        {
            "role": "system",
            "content": enhanced_system_structured,
        },  # ✅ Now includes learned patterns + caching
        {
            "role": "system",
            "content": dynamic_context_message,
        },  # Dynamic content not cached
        *conversation_messages,
        {"role": "user", "content": tool_call_prompt},
    ]

    try:
        # Execute tool call
        model_with_tools = model.bind_tools(TOOLS)
        tool_response = await model_with_tools.ainvoke(
            tool_messages,
            tool_choice={
                "type": "function",
                "function": {"name": current_step.tool_name},
            },
        )

        # ✅ ADD LOGGING
        if tool_response.tool_calls:
            logger.info(f"🔧 [ACT] Tool execution requested:")
            for tc in tool_response.tool_calls:
                tool_name = tc.get("name", "unknown")
                tool_args = tc.get("args", {})
                args_str = json.dumps(tool_args, indent=2)[:200]
                logger.info(f"🔧 [ACT]   - {tool_name}")
                logger.info(f"🔧 [ACT]     Args: {args_str}")
        else:
            logger.warning(
                f"🔧 [ACT] No tool calls generated (expected {current_step.tool_name})"
            )

        # TIER 0: Check for micro-retry opportunity
        # We need to wait for the tool to actually execute to check its result
        # This check will happen in the assessment phase, but we store the tool response

        # Combine narration with tool call
        combined_response = AIMessage(
            content=llm_generated_narration,
            tool_calls=tool_response.tool_calls,
            additional_kwargs=tool_response.additional_kwargs,
            response_metadata=tool_response.response_metadata,
            id=tool_response.id,
        )

        # ✅ FIX: Persist memory state before potential interrupt (file approval)
        if state.memory and state.memory.auto_persist:
            try:
                await state.memory._persist_to_database()
                logger.info("🧠 [ACT] Pre-persisted memory before tool execution")
            except Exception as e:
                logger.warning(f"🧠 [ACT] Pre-persistence failed: {e}")

        return {
            "plan": in_progress_plan,
            "messages": [combined_response],
            "ui": [in_progress_ui_msg],
            "total_tool_calls": state.total_tool_calls
            + (len(tool_response.tool_calls) if tool_response.tool_calls else 0),
        }

    except Exception as e:
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

        # Emit error plan UI update (reuse existing ID)
        error_ui_msg = create_plan_ui_message(
            updated_plan, "", ui_message_id=state.plan_ui_message_id
        )

        return {
            "plan": updated_plan,
            "messages": [AIMessage(content=error_narration)],
            "ui": [error_ui_msg],
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

    for phrase in generic_phrases:
        if phrase in narration_lower:
            return True

    if len(narration_lower) < 20:
        return True

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
    base_narrations = {
        "search_project": f"Querying indexed project data for: {current_step.description}",
        "code_snippets": f"Searching scripts semantically for: {current_step.description}",
        "unity_docs": f"Searching Unity documentation for: {current_step.description}",
        "web_search": f"Searching Unity/Unreal resources and tutorials for: {current_step.description}",
    }

    base_narration = base_narrations.get(
        current_step.tool_name,
        f"Executing {current_step.tool_name} for: {current_step.description}",
    )

    if retry_count > 0:
        retry_prefixes = {
            1: "Trying a different approach - ",
            2: "Let me retry this with adjusted parameters - ",
            3: "Making another attempt - ",
        }
        prefix = retry_prefixes.get(retry_count, f"Attempt #{retry_count + 1} - ")
        base_narration = prefix + base_narration.lower()

    step_info = (
        f" (Step {step_context['step_index'] + 1} of {step_context['total_steps']})"
    )
    return base_narration + step_info
