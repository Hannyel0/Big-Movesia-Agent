"""Planning node that uses project context from runtime metadata.

This demonstrates how ANY node can access the project context that was
passed via config and stored in runtime_metadata by the classify node.
"""

from __future__ import annotations

import json
from datetime import datetime, UTC
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field
from langgraph.runtime import Runtime

from react_agent.context import Context
# Note: We don't use get_cacheable_planning_prompt() here because plan.py builds
# a custom dynamic prompt with tool metadata, project context, and memory injection.
# Instead, we manually apply cache control to the custom prompt below.
from react_agent.prompts import PLANNING_PROMPT
from react_agent.state import (
    ExecutionPlan,
    PlanStep,
    State,
    StepStatus,
    StructuredExecutionPlan,
    StructuredPlanStep,
    ToolName,
)
from react_agent.tools import TOOLS, TOOL_METADATA
from react_agent.narration import NarrationEngine
from react_agent.utils import get_message_text, get_model, is_anthropic_model
from react_agent.memory import inject_memory_into_prompt


# Build tools description once at module level for caching
def _build_comprehensive_tools_description() -> str:
    """Build comprehensive tools description with usage examples."""
    tools_info = []
    for tool in TOOLS:
        tool_info = f"**{tool.name}**: {tool.description}\n"
        
        if hasattr(tool, 'name') and tool.name in TOOL_METADATA:
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
        elif tool.name in ["read_file", "write_file", "modify_file", "delete_file", "move_file"]:
            if tool.name == "read_file":
                tool_info += "   - Example uses: Read existing scripts, inspect file contents\n"
            elif tool.name == "write_file":
                tool_info += "   - Example uses: Create new scripts, generate code files\n"
            elif tool.name == "modify_file":
                tool_info += "   - Example uses: Update existing code, fix bugs, add features\n"
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
            if any(word in content.lower() for word in ["beginner", "new to", "learning", "tutorial"]):
                context_parts.append("User appears to be learning/beginner level")
            if any(word in content.lower() for word in ["advanced", "experienced", "complex", "sophisticated"]):
                context_parts.append("User appears to be experienced")
            if "error" in content.lower() or "problem" in content.lower():
                context_parts.append("User is troubleshooting an issue")
    
    return "; ".join(context_parts) if context_parts else "No specific context detected"


def _build_intelligent_planning_context(user_message: str, conversation_context: str) -> str:
    """Build rich context for intelligent planning."""
    
    return f"""PLANNING REQUEST ANALYSIS:
User Request: "{user_message}"
Conversation Context: {conversation_context}

INTELLIGENT PLANNING GUIDELINES:
1. **Analyze the specific request** - Don't use templates, understand what the user actually needs
2. **Consider user expertise** - Adapt complexity based on their apparent skill level
3. **Optimize for efficiency** - Sometimes 2 steps are better than 5, sometimes 6 steps are needed
4. **Think about dependencies** - Some steps naturally depend on others, some can be parallel
5. **Consider alternatives** - There might be multiple valid approaches, choose the best one
6. **Factor in context** - Existing project? Learning exercise? Debugging? Each needs different approach

TOOL SELECTION REASONING:
- Don't always start with search - sometimes you can go straight to implementation
- Don't always get project info first - depends on what you're doing
- Don't always compile at the end - sometimes it's not needed
- Think about what the user ACTUALLY needs, not what the template says

FLEXIBLE APPROACH EXAMPLES:
- Simple script request: Might just need get_script_snippets → write_file
- Complex system: Might need search → get_project_info → multiple create_asset → write_file → compile_and_test
- Debugging: Might need get_project_info → compile_and_test → search → write_file
- Learning request: Might need search → get_script_snippets with detailed explanations

CREATE A PLAN THAT MAKES SENSE FOR THIS SPECIFIC REQUEST, NOT A GENERIC TEMPLATE."""


async def plan(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Create execution plan using LLM - now with project context awareness."""
    
    # ✅ NEW: Access project context from runtime_metadata (stored by classify node)
    project_id = state.runtime_metadata.get("project_id", "")
    project_root = state.runtime_metadata.get("project_root", "")
    project_name = state.runtime_metadata.get("project_name", "")
    unity_version = state.runtime_metadata.get("unity_version", "")
    
    print(f"\n{'='*60}")
    print(f"📋 [Plan] Creating plan with project context:")
    print(f"  Project: {project_name} (ID: {project_id})")
    print(f"  Root: {project_root}")
    print(f"  Unity: {unity_version}")
    print(f"{'='*60}\n")
    
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
    planning_context = _build_intelligent_planning_context(user_message, conversation_context)
    
    # TRULY INTELLIGENT PLANNING PROMPT - No templates, pure reasoning
    intelligent_planning_request = f"""{planning_context}

Now create an intelligent, context-aware plan. Consider:

SPECIFIC REQUEST ANALYSIS:
- What exactly is the user trying to accomplish?
- What's the most efficient path to get there?
- What level of detail/explanation do they need?
- Are there any shortcuts or optimizations possible?

TOOL STRATEGY REASONING:
- Which tools are essential vs nice-to-have for this request?
- What's the logical dependency chain?
- Can any steps be combined or eliminated?
- Should I prioritize speed, thoroughness, or education?

CONTEXTUAL ADAPTATIONS:
- Does this user seem to need detailed guidance or quick results?
- Is this a one-off task or part of a larger learning journey?
- Are there project-specific considerations I should account for?

Create a smart, efficient plan that solves the user's actual need rather than following a rigid template."""

    # ✅ NEW: Enhanced planning prompt with project context
    project_context_str = f"""
Current Project Context:
- Project Name: {project_name or "Not specified"}
- Unity Version: {unity_version or "Not specified"}
- Project Root: {project_root or "Not specified"}

Consider this context when creating your plan. For example:
- If Unity version is known, ensure compatibility with that version
- If project structure is available, reference existing assets
- Plan steps should be appropriate for the project setup
"""

    # CORRECTED SYSTEM CONTENT for intelligent planning
    base_system_content = f"""You are an expert Unity/game development planning assistant with access to these tools:

{COMPREHENSIVE_TOOLS_DESCRIPTION}

{project_context_str if project_id else ""}

TOOL USAGE CLARIFICATION:
**search_project**: Query indexed Unity project database using natural language
- Use to find assets, GameObjects, components, and dependencies in the project
- Use to understand project structure and existing implementations
- Use to check what's already available before creating new content

**code_snippets**: Semantic search through C# scripts by functionality
- Use to find existing code that does what you need
- Use to discover implementations and patterns in the project
- Use when you need to understand how something is already coded

**unity_docs**: Search local Unity documentation with semantic RAG
- Use for Unity API references and feature documentation
- Use to learn about Unity classes, methods, and components
- Best for: API lookup, scripting examples, Unity feature documentation

**read_file**: Read file contents safely
- Use to read existing scripts and understand current implementations
- Use to inspect file contents before making changes
- No approval required

**write_file**: Create/write files (requires approval)
- Use to create new script files
- Use to generate new implementations
- Requires human approval before execution

**modify_file**: Modify existing files (requires approval)
- Use to update existing scripts
- Use for surgical code edits
- Requires human approval before execution

**delete_file**: Delete files (requires approval)
- Use to remove old or unused scripts
- Use to clean up project files
- Requires human approval before execution

**move_file**: Move/rename files (requires approval)
- Use to reorganize project structure
- Use to rename scripts
- Requires human approval before execution

**web_search**: Research external Unity documentation, tutorials, best practices
- Use for finding implementation approaches for new features
- Use for troubleshooting and learning about Unity systems
- Use when you need information not available in the project

INTELLIGENT PLANNING PRINCIPLES:
- **To understand existing project**: search_project → discover assets and structure
- **To find existing code**: code_snippets → locate relevant implementations
- **To learn Unity API/features**: unity_docs → get official Unity documentation
- **To learn new approaches**: web_search → research implementation methods
- **To build on existing code**: code_snippets → read_file → modify_file
- **To create new features**: unity_docs OR web_search → write_file

PLANNING EXAMPLES:
- "Fix player movement" → code_snippets → find movement code → modify_file
- "Add grass physics" → unity_docs → learn physics API → write_file
- "Improve existing UI" → search_project → find UI assets → code_snippets → modify_file
- "Create AI enemy" → unity_docs → learn AI components → write_file
- "How do Collider2D work?" → unity_docs → get API reference

CREATE PLANS BASED ON WHETHER YOU NEED TO READ EXISTING CODE OR RESEARCH NEW SOLUTIONS.

Remember: Every step must specify a concrete tool to use. No generic steps allowed."""
    
    # ✅ MEMORY: Inject memory context if available
    intelligent_system_content = await inject_memory_into_prompt(
        base_prompt=base_system_content,
        state=state,
        include_patterns=True,
        include_episodes=True
    )

    # ✅ CACHING: Convert to cacheable format if enabled AND using Anthropic
    # OpenAI models cache automatically, no explicit markers needed
    cache_enabled = getattr(context, 'enable_prompt_cache', True)
    using_anthropic = is_anthropic_model(context.planning_model or context.model)

    if cache_enabled and using_anthropic:
        # Use structured content format with cache control for Anthropic models only
        system_content_structured = [
            {
                "type": "text",
                "text": intelligent_system_content,
                "cache_control": {"type": "ephemeral"}
            }
        ]
    else:
        # Use simple string format for OpenAI (automatic caching) or when cache disabled
        system_content_structured = intelligent_system_content

    # Structure messages for intelligent planning
    messages = [
        {"role": "system", "content": system_content_structured},
        {"role": "user", "content": intelligent_planning_request}
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
                dependencies=step_data.dependencies
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
            }
        )
        
        # ✅ MEMORY: Store plan in memory (manager handles both episodic and working memory)
        if state.memory:
            state.memory.add_plan({
                "goal": plan.goal,
                "steps": [{"description": s.description, "tool": s.tool_name} for s in steps]
            })
            print(f"🧠 [Plan] Plan stored in memory: {len(steps)} steps")
        
        # CREATE CONTEXTUAL PLANNING NARRRATION
        planning_narration = _create_intelligent_planning_narration(plan, user_message, conversation_context)
        
        return {
            "plan": plan,
            "step_index": 0,
            "retry_count": 0,
            "messages": [AIMessage(content=planning_narration)]
        }
    
    except Exception as e:
        # Fallback to minimal intelligent planning
        fallback_steps = _create_minimal_intelligent_plan(user_message, conversation_context)
        fallback_plan = ExecutionPlan(
            goal=user_message,
            steps=fallback_steps,
            metadata={"planning_mode": "intelligent_fallback"}
        )
        
        fallback_narration = f"I'll approach this intelligently based on your specific need: **{user_message}**"
        
        return {
            "plan": fallback_plan,
            "step_index": 0,
            "retry_count": 0,
            "messages": [AIMessage(content=fallback_narration)]
        }


def _create_intelligent_planning_narration(plan: ExecutionPlan, user_message: str, context: str) -> str:
    """Create contextual planning narration with proper markdown formatting."""
    
    narration = f"## 🎯 Development Plan\n\nI'll help you with **{user_message}** using a tailored approach.\n\n"
    
    # Explain the reasoning based on plan characteristics
    step_count = len(plan.steps)
    
    if step_count == 1:
        narration += "### Approach\n\nThis looks like a straightforward request that I can handle directly:\n\n"
    elif step_count == 2:
        narration += "### Approach\n\nI'll take a focused two-step approach:\n\n"
    elif step_count <= 4:
        narration += f"### Approach\n\nI've designed a **{step_count}-step plan** that balances efficiency with thoroughness:\n\n"
    else:
        narration += f"### Approach\n\nThis requires a comprehensive **{step_count}-step approach** to ensure quality:\n\n"
    
    # List the steps with proper markdown formatting
    for i, step in enumerate(plan.steps, 1):
        tool_emoji = {
            "search_project": "🔍",
            "code_snippets": "📝",
            "unity_docs": "📚",
            "read_file": "📖",
            "write_file": "✏️",
            "modify_file": "🔧",
            "delete_file": "🗑️",
            "move_file": "📦",
            "web_search": "🌐"
        }.get(step.tool_name, "🛠️")
        
        narration += f"{i}. {tool_emoji} **{step.tool_name}**: {step.description}\n"
        
        # Add contextual reasoning for why this step makes sense
        if step.tool_name == "web_search" and i > 1:
            narration += "   *Research needed for this specific implementation*\n"
        elif step.tool_name == "search_project" and i > 2:
            narration += "   *Checking your project setup for compatibility*\n"
        elif step.tool_name == "code_snippets" and "web_search" not in [s.tool_name for s in plan.steps[:i-1]]:
            narration += "   *I have the code patterns you need*\n"
        elif step.tool_name in ["write_file", "modify_file", "delete_file", "move_file"] and step_count > 3:
            narration += "   *Ensuring everything integrates properly*\n"
    
    narration += "\n"
    
    # Add context-aware conclusion with proper markdown
    if "beginner" in context.lower():
        narration += "### 💡 Note\n\nI'll provide detailed explanations since you're learning.\n"
    elif "experienced" in context.lower():
        narration += "### ⚡ Note\n\nI'll focus on efficient implementation given your experience.\n"
    elif "troubleshooting" in context.lower():
        narration += "### 🔧 Note\n\nI'll prioritize identifying and fixing the issue quickly.\n"
    
    return narration


def _create_minimal_intelligent_plan(user_message: str, context: str) -> List[PlanStep]:
    """Create a minimal intelligent plan when structured planning fails."""
    
    message_lower = user_message.lower()
    
    # INTELLIGENT analysis rather than rigid templates
    
    # Direct information requests
    if any(starter in message_lower for starter in ["what is", "how does", "explain", "tell me about"]):
        return [
            PlanStep(
                description=f"Research and provide comprehensive information about: {user_message}",
                tool_name="web_search",
                success_criteria="Found relevant, detailed information to answer the question"
            )
        ]
    
    # Quick implementation requests
    if any(word in message_lower for word in ["simple", "basic", "quick"]) and "script" in message_lower:
        return [
            PlanStep(
                description="Find existing code patterns for the requested functionality",
                tool_name="code_snippets", 
                success_criteria="Retrieved appropriate code examples"
            ),
            PlanStep(
                description="Create the script file with the implementation",
                tool_name="write_file",
                success_criteria="Successfully created working script file",
                dependencies=[0]
            )
        ]
    
    # Debugging/problem-solving requests
    if any(word in message_lower for word in ["error", "problem", "fix", "debug", "broken"]):
        return [
            PlanStep(
                description="Analyze current project state to identify the issue",
                tool_name="search_project",
                success_criteria="Identified project configuration and potential issues"
            ),
            PlanStep(
                description="Find existing code that might be causing the problem",
                tool_name="code_snippets",
                success_criteria="Located relevant code implementations", 
                dependencies=[0]
            ),
            PlanStep(
                description="Research solutions for the identified problems",
                tool_name="web_search",
                success_criteria="Found relevant troubleshooting information",
                dependencies=[1]
            )
        ]
    
    # Complex creation requests  
    if any(word in message_lower for word in ["complete", "full", "comprehensive", "system"]):
        return [
            PlanStep(
                description="Research current best practices and approaches",
                tool_name="web_search", 
                success_criteria="Found comprehensive implementation guidance"
            ),
            PlanStep(
                description="Understand project context and requirements",
                tool_name="search_project",
                success_criteria="Analyzed project setup and compatibility requirements"
            ),
            PlanStep(
                description="Find existing code examples and patterns",
                tool_name="code_snippets",
                success_criteria="Retrieved comprehensive code patterns",
                dependencies=[0, 1]
            ),
            PlanStep(
                description="Implement the complete solution",
                tool_name="write_file", 
                success_criteria="Created working implementation file",
                dependencies=[2]
            ),
            PlanStep(
                description="Validate the implementation with project queries", 
                tool_name="search_project",
                success_criteria="Verified solution integrates correctly with project",
                dependencies=[3]
            )
        ]
    
    # Default intelligent approach
    return [
        PlanStep(
            description=f"Analyze and understand the specific requirements for: {user_message}",
            tool_name="search_project",
            success_criteria="Gathered relevant context and project information"
        ),
        PlanStep(
            description="Implement the requested functionality efficiently", 
            tool_name="code_snippets" if "code" in message_lower or "script" in message_lower else "web_search",
            success_criteria="Provided working solution for the request",
            dependencies=[0]
        )
    ]