"""Planning node that uses project context from runtime metadata.

This demonstrates how ANY node can access the project context that was
passed via config and stored in runtime_metadata by the classify node.
"""

from __future__ import annotations

import uuid
import logging
from datetime import datetime, UTC
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from react_agent.context import Context

from react_agent.state import (
    ExecutionPlan,
    PlanStep,
    State,
    StructuredExecutionPlan,
)
from react_agent.tools import TOOLS, TOOL_METADATA
from react_agent.narration import NarrationEngine
from react_agent.utils import get_message_text, get_model, is_anthropic_model
from react_agent.memory import inject_memory_into_prompt
from react_agent.ui_messages import create_plan_ui_message


# Set up logging
logger = logging.getLogger(__name__)


# Build tools description once at module level for caching
def _build_comprehensive_tools_description() -> str:
    """Build comprehensive tools description with usage examples."""
    tools_info = []
    for tool in TOOLS:
        tool_info = f"**{tool.name}**: {tool.description}\n"

        if hasattr(tool, "name") and tool.name in TOOL_METADATA:
            metadata = TOOL_METADATA[tool.name]
            tool_info += f"   - Best for: {', '.join(metadata['best_for'])}\n"
            tool_info += f"   - Reliability: {metadata['reliability']}, Cost: {metadata['cost']}\n"

        # Add practical usage examples
        if tool.name == "search_project":
            tool_info += "   - Example uses: Find assets by name, query GameObject hierarchy, check component usage\n"
        elif tool.name == "code_snippets":
            tool_info += "   - Example uses: Find movement code by functionality, locate UI patterns, discover physics implementations\n"
        elif tool.name == "unity_docs":
            tool_info += "   - Example uses: Unity API reference, Collider2D documentation, particle system features\n"
        elif tool.name in [
            "read_file",
            "write_file",
            "modify_file",
            "delete_file",
            "move_file",
        ]:
            if tool.name == "read_file":
                tool_info += (
                    "   - Example uses: Read existing scripts, inspect file contents\n"
                )
            elif tool.name == "write_file":
                tool_info += (
                    "   - Example uses: Create new scripts, generate code files\n"
                )
            elif tool.name == "modify_file":
                tool_info += (
                    "   - Example uses: Update existing code, fix bugs, add features\n"
                )
            elif tool.name == "delete_file":
                tool_info += "   - Example uses: Remove old scripts, clean up files\n"
            elif tool.name == "move_file":
                tool_info += "   - Example uses: Reorganize scripts, rename files\n"
        elif tool.name == "web_search":
            tool_info += "   - Example uses: Research Unity patterns, find tutorials, troubleshoot errors\n"

        tools_info.append(tool_info)

    return "\n".join(tools_info)


# Static tools description built once
COMPREHENSIVE_TOOLS_DESCRIPTION = _build_comprehensive_tools_description()

# Initialize narration components
narration_engine = NarrationEngine()


def _extract_conversation_context(state: State) -> str:
    """Extract relevant context from the conversation history."""
    context_parts = []

    # Look for previous attempts or mentions
    for msg in state.messages[-5:]:  # Last 5 messages for context
        if isinstance(msg, HumanMessage):
            content = get_message_text(msg)
            # Extract context clues
            if "my project" in content.lower():
                context_parts.append("User has an existing project")
            if any(
                word in content.lower()
                for word in ["beginner", "new to", "learning", "tutorial"]
            ):
                context_parts.append("User appears to be learning/beginner level")
            if any(
                word in content.lower()
                for word in ["advanced", "experienced", "complex", "sophisticated"]
            ):
                context_parts.append("User appears to be experienced")
            if "error" in content.lower() or "problem" in content.lower():
                context_parts.append("User is troubleshooting an issue")

    return "; ".join(context_parts) if context_parts else "No specific context detected"


def _build_intelligent_planning_context(
    user_message: str, conversation_context: str
) -> str:
    """Build compressed context for intelligent planning (~80 tokens)."""

    return f"""Request: "{user_message}"
Context: {conversation_context}

Rules:
- Match request needs, not templates
- Adapt to user skill level
- Optimize step count (2-6 typical)
- Specify dependencies
- Choose best tools for task

Examples:
- Simple script: code_snippets → write_file
- Complex system: search → get_project → multiple creates → compile
- Debug: get_project → compile → search → fix"""


async def plan(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Create execution plan using LLM - now with project context awareness.
    
    Token Optimization: Reduced from ~1,500 tokens to ~400 tokens per call (~73% reduction)
    - Planning context: 500 → 80 tokens
    - Planning request: 800 → 150 tokens  
    - System content: 600 → 200 tokens
    - Project context: 100 → 50 tokens
    """

    # NEW: Access project context from runtime_metadata (stored by classify node)
    project_id = state.runtime_metadata.get("project_id", "")
    project_root = state.runtime_metadata.get("project_root", "")
    project_name = state.runtime_metadata.get("project_name", "")
    unity_version = state.runtime_metadata.get("unity_version", "")

    print(f"\n{'=' * 60}")
    print(f"[Plan] Creating plan with project context:")
    print(f"  Project: {project_name} (ID: {project_id})")
    print(f"  Root: {project_root}")
    print(f"  Unity: {unity_version}")
    print(f"{'=' * 60}\n")

    context = runtime.context
    model = get_model(context.planning_model or context.model)

    # Extract user request
    user_message = None
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            user_message = get_message_text(msg)
            break

    if not user_message:
        raise ValueError("No user request found in messages")

    # Extract conversation context for intelligent planning
    conversation_context = _extract_conversation_context(state)

    # Build intelligent planning context
    planning_context = _build_intelligent_planning_context(
        user_message, conversation_context
    )

    # Optimized planning request (~150 tokens vs 800)
    intelligent_planning_request = f"""{planning_context}

Create optimal plan:

Analysis:
- User intent & efficiency path
- Required vs optional tools
- Step dependencies & combinations

Adapt to:
- User skill level (guidance vs speed)
- Task type (one-off vs learning)
- Project context

Deliver smart, efficient plan for actual need."""

    # Optimized project context (~50 tokens vs 100)
    project_context_str = f"""
Project: {project_name or "N/A"} | Unity: {unity_version or "N/A"} | Root: {project_root or "N/A"}
Ensure version compatibility and reference existing assets.
"""

    # Optimized system content (~200 tokens vs 600)
    base_system_content = f"""Unity dev planning assistant. Tools: {COMPREHENSIVE_TOOLS_DESCRIPTION}

{project_context_str if project_id else ""}

Tool patterns:
- Understand project: search_project
- Find code: code_snippets
- Learn Unity API: unity_docs
- Research methods: web_search
- Read/inspect: read_file
- Create: write_file (approval)
- Update: modify_file (approval)
- Clean: delete_file/move_file (approval)

Common flows:
- Fix code: code_snippets → modify_file
- New feature: unity_docs/web_search → write_file
- Improve existing: search_project → code_snippets → modify_file

Every step needs specific tool. No generic steps."""

    # MEMORY: Inject memory context if available
    intelligent_system_content = await inject_memory_into_prompt(
        base_prompt=base_system_content,
        state=state,
        include_patterns=True,
        include_episodes=True,
    )

    # CACHING: Convert to cacheable format if enabled AND using Anthropic
    # OpenAI models cache automatically, no explicit markers needed
    cache_enabled = getattr(context, "enable_prompt_cache", True)
    using_anthropic = is_anthropic_model(context.planning_model or context.model)

    if cache_enabled and using_anthropic:
        # Use structured content format with cache control for Anthropic models only
        system_content_structured = [
            {
                "type": "text",
                "text": intelligent_system_content,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    else:
        # Use simple string format for OpenAI (automatic caching) or when cache disabled
        system_content_structured = intelligent_system_content

    # Structure messages for intelligent planning
    messages = [
        {"role": "system", "content": system_content_structured},
        {"role": "user", "content": intelligent_planning_request},
    ]

    try:
        # Use structured output for reliable planning but with intelligent reasoning
        structured_model = model.with_structured_output(StructuredExecutionPlan)
        structured_response = await structured_model.ainvoke(messages)

        # Convert structured response to internal format
        steps = []
        for step_data in structured_response.steps:
            step = PlanStep(
                description=step_data.description,
                success_criteria=step_data.success_criteria,
                tool_name=step_data.tool_name,
                dependencies=step_data.dependencies,
            )
            steps.append(step)

        # When creating the final plan, you can also store project context in plan metadata:
        plan = ExecutionPlan(
            goal=structured_response.goal,
            steps=steps,
            metadata={
                "planning_mode": "intelligent",
                "context_considered": conversation_context,
                "original_request": user_message,
                "project_id": project_id,
                "project_name": project_name,
                "unity_version": unity_version,
                "planned_at": datetime.now(UTC).isoformat(),
            },
        )

        # MEMORY: Store plan in memory (manager handles both episodic and working memory)
        if state.memory:
            state.memory.add_plan(
                {
                    "goal": plan.goal,
                    "steps": [
                        {"description": s.description, "tool": s.tool_name}
                        for s in steps
                    ],
                }
            )
            print(f"[Plan] Plan stored in memory: {len(steps)} steps")

        # Create concise narration (detailed plan shown in UI component)
        step_count = len(plan.steps)
        planning_narration = f"I've created a {step_count}-step plan to help you. Beginning execution now."

        # Create stable message ID first
        msg_id = str(uuid.uuid4())

        # Create AI message with explicit ID
        msg = AIMessage(content=planning_narration, id=msg_id)

        # Emit UIMessage for plan visualization with the same stable ID
        # Reuse existing plan_ui_message_id if replanning, otherwise create new
        plan_ui_msg = create_plan_ui_message(
            plan, msg_id, ui_message_id=state.plan_ui_message_id
        )

        result = {
            "plan": plan,
            "plan_ui_message_id": plan_ui_msg.id,  # ✅ CRITICAL: Store the ID
            "step_index": 0,
            "retry_count": 0,
            "messages": [msg],
            "ui": [plan_ui_msg],
        }

        return result

    except Exception:
        # Fallback to minimal intelligent planning
        fallback_steps = _create_minimal_intelligent_plan(
            user_message, conversation_context
        )
        fallback_plan = ExecutionPlan(
            goal=user_message,
            steps=fallback_steps,
            metadata={"planning_mode": "intelligent_fallback"},
        )

        fallback_narration = f"I'll approach this intelligently based on your specific need: **{user_message}**"

        # Create stable message ID first
        fallback_msg_id = str(uuid.uuid4())

        # Create AI message with explicit ID
        fallback_msg = AIMessage(content=fallback_narration, id=fallback_msg_id)

        # Emit UIMessage for fallback plan visualization
        # Reuse existing plan_ui_message_id if replanning, otherwise create new
        fallback_plan_ui_msg = create_plan_ui_message(
            fallback_plan, fallback_msg_id, ui_message_id=state.plan_ui_message_id
        )

        result = {
            "plan": fallback_plan,
            "plan_ui_message_id": fallback_plan_ui_msg.id,  # ✅ CRITICAL: Store the ID
            "step_index": 0,
            "retry_count": 0,
            "messages": [fallback_msg],
            "ui": [fallback_plan_ui_msg],
        }

        return result


def _create_minimal_intelligent_plan(user_message: str, context: str) -> List[PlanStep]:
    """Create a minimal intelligent plan when structured planning fails."""

    message_lower = user_message.lower()

    # INTELLIGENT analysis rather than rigid templates

    # Direct information requests
    if any(
        starter in message_lower
        for starter in ["what is", "how does", "explain", "tell me about"]
    ):
        return [
            PlanStep(
                description=f"Research and provide comprehensive information about: {user_message}",
                tool_name="web_search",
                success_criteria="Found relevant, detailed information to answer the question",
            )
        ]

    # Quick implementation requests
    if (
        any(word in message_lower for word in ["simple", "basic", "quick"])
        and "script" in message_lower
    ):
        return [
            PlanStep(
                description="Find existing code patterns for the requested functionality",
                tool_name="code_snippets",
                success_criteria="Retrieved appropriate code examples",
            ),
            PlanStep(
                description="Create the script file with the implementation",
                tool_name="write_file",
                success_criteria="Successfully created working script file",
                dependencies=[0],
            ),
        ]

    # Debugging/problem-solving requests
    if any(
        word in message_lower for word in ["error", "problem", "fix", "debug", "broken"]
    ):
        return [
            PlanStep(
                description="Analyze current project state to identify the issue",
                tool_name="search_project",
                success_criteria="Identified project configuration and potential issues",
            ),
            PlanStep(
                description="Find existing code that might be causing the problem",
                tool_name="code_snippets",
                success_criteria="Located relevant code implementations",
                dependencies=[0],
            ),
            PlanStep(
                description="Research solutions for the identified problems",
                tool_name="web_search",
                success_criteria="Found relevant troubleshooting information",
                dependencies=[1],
            ),
        ]

    # Complex creation requests
    if any(
        word in message_lower
        for word in ["complete", "full", "comprehensive", "system"]
    ):
        return [
            PlanStep(
                description="Research current best practices and approaches",
                tool_name="web_search",
                success_criteria="Found comprehensive implementation guidance",
            ),
            PlanStep(
                description="Understand project context and requirements",
                tool_name="search_project",
                success_criteria="Analyzed project setup and compatibility requirements",
            ),
            PlanStep(
                description="Find existing code examples and patterns",
                tool_name="code_snippets",
                success_criteria="Retrieved comprehensive code patterns",
                dependencies=[0, 1],
            ),
            PlanStep(
                description="Implement the complete solution",
                tool_name="write_file",
                success_criteria="Created working implementation file",
                dependencies=[2],
            ),
            PlanStep(
                description="Validate the implementation with project queries",
                tool_name="search_project",
                success_criteria="Verified solution integrates correctly with project",
                dependencies=[3],
            ),
        ]

    # Default intelligent approach
    return [
        PlanStep(
            description=f"Analyze and understand the specific requirements for: {user_message}",
            tool_name="search_project",
            success_criteria="Gathered relevant context and project information",
        ),
        PlanStep(
            description="Implement the requested functionality efficiently",
            tool_name="code_snippets"
            if "code" in message_lower or "script" in message_lower
            else "web_search",
            success_criteria="Provided working solution for the request",
            dependencies=[0],
        ),
    ]
