"""Complexity classification node with continuation awareness."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional
import logging

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field
import json

from react_agent.context import Context
from react_agent.state import State
from react_agent.utils import get_message_text, get_model
from react_agent.memory import (
    initialize_memory_system, 
    extract_entities_from_request, 
    extract_topics_from_request,
    inject_memory_into_prompt
)

logger = logging.getLogger(__name__)


class ComplexityAssessment(BaseModel):
    """Structured output for complexity classification."""
    complexity_level: Literal["direct", "simple_plan", "complex_plan"] = Field(
        description="Classification of request complexity"
    )
    reasoning: str = Field(
        description="Brief explanation of why this complexity level was chosen"
    )
    suggested_approach: str = Field(
        description="Recommended execution approach for this request"
    )
    user_narration: str = Field(
        description="User-facing description of what you'll do to help them (no internal details)"
    )
    confidence: float = Field(
        default=0.8, 
        ge=0.0, 
        le=1.0, 
        description="Confidence in the complexity assessment"
    )


def _detect_continuation_context(state: State) -> Optional[Dict[str, Any]]:
    """
    Detect if the current request is a continuation from previous interactions.
    Returns context about the previous interaction if detected.
    """
    if not state.memory or not hasattr(state.memory, 'working_memory'):
        return None
    
    recent_tools = state.memory.working_memory.recent_tool_results
    if not recent_tools:
        return None
    
    # Get the most recent tool result
    latest_tool = recent_tools[-1]
    
    # Get the last AI message (what the user saw)
    last_ai_message = None
    for msg in reversed(state.messages[-5:]):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            last_ai_message = get_message_text(msg)
            break
    
    # Get current user message
    current_user_msg = None
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            current_user_msg = get_message_text(msg).lower()
            break
    
    if not current_user_msg:
        return None
    
    # Check if this looks like a continuation
    continuation_patterns = [
        "which", "what about", "how about", "the biggest", "the largest", 
        "the smallest", "the first", "the second", "the last", "show me",
        "tell me more", "more details", "more info", "yes", "yeah"
    ]
    
    is_continuation = (
        len(current_user_msg.split()) <= 10 and  # Short request
        any(pattern in current_user_msg for pattern in continuation_patterns)
    )
    
    if not is_continuation:
        return None
    
    return {
        "previous_message": last_ai_message or "Previous results",
        "tool_result": {
            "tool_name": latest_tool["tool_name"],
            "result": latest_tool["result"],
            "summary": latest_tool["summary"]
        }
    }


async def classify(
    state: State, 
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> Dict[str, Any]:
    """Classify request complexity with continuation awareness."""
    
    # Memory initialization
    memory_updates = {}
    if not state.memory:
        memory_updates = await initialize_memory_system(state, config)
    
    # Extract project context
    configurable = config.get("configurable", {})
    project_id = configurable.get("project_id", "")
    project_root = configurable.get("project_root", "")
    project_name = configurable.get("project_name", "")
    unity_version = configurable.get("unity_version", "")
    sqlite_path = configurable.get("sqlite_path", "")
    submitted_at = configurable.get("submitted_at", "")
    
    context = runtime.context
    model = get_model(context.model)
    
    # Extract user request
    user_request = None
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            user_request = get_message_text(msg)
            break
    
    if not user_request:
        return {
            "runtime_metadata": {
                **state.runtime_metadata,
                "complexity_level": "complex_plan",
                "complexity_reasoning": "No clear request found",
                "project_id": project_id,
                "project_root": project_root,
                "unity_version": unity_version,
                "sqlite_path": sqlite_path,
            },
            **memory_updates
        }
    
    # Get memory reference
    memory_to_use = memory_updates.get("memory") or state.memory
    
    # CRITICAL: Force reload of working memory BEFORE continuation detection
    # This ensures we can detect continuations even if memory was just initialized
    if memory_to_use and hasattr(memory_to_use, 'working_memory'):
        wm = memory_to_use.working_memory
        if len(wm.recent_tool_results) == 0 and memory_to_use.db_path and memory_to_use.db_path.exists():
            try:
                # Manually reload using the same logic from get_memory_context()
                import asyncio
                import sqlite3
                import json as json_module
                
                def _reload_tools():
                    conn = sqlite3.connect(str(memory_to_use.db_path), timeout=5.0)
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT tool_name, query, result, summary, timestamp
                        FROM memory_working
                        WHERE session_id = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                    """, (memory_to_use.session_id, wm.max_tool_results))
                    
                    tools = []
                    for row in cursor.fetchall():
                        try:
                            tool_result = {
                                "tool_name": row[0],
                                "query": row[1],
                                "result": json_module.loads(row[2]),
                                "summary": row[3],
                                "timestamp": row[4]
                            }
                            tools.append(tool_result)
                        except Exception:
                            pass
                    
                    conn.close()
                    return tools
                
                loaded_tools = await asyncio.to_thread(_reload_tools)
                for tool in loaded_tools:
                    wm.recent_tool_results.append(tool)
            except Exception:
                pass
    
    # Detect continuation context AFTER ensuring memory is loaded
    continuation_context = None
    if memory_to_use:
        continuation_context = _detect_continuation_context(state)
    
    # Base classification prompt
    base_classification_prompt = """You are a Unity/Unreal Engine development complexity assessor with access to production tools.

**Available Production Tools:**
- **search_project**: Query indexed project database
- **code_snippets**: Semantic search through C# scripts
- **file_operation**: Safe file I/O with validation
- **web_search**: Research Unity documentation and tutorials

**CRITICAL FOR CONTINUATION REQUESTS:**
When the user's request references previous results or context (like "which is the biggest one?", "show me more", "the first one"), this is a DIRECT action that should:
1. Use the existing context from previous tool results
2. Analyze or present that data differently
3. NOT require new tool executions or planning

**DIRECT**: Single-step responses
- Project data queries: "What assets do I have?", "List my scripts"
- CONTINUATION QUERIES: "Which is the biggest?", "Show me more details", "The first one"
- Informational queries: "What is a prefab?", "How do controllers work?"
- Follow-up analysis of existing results

**SIMPLE_PLAN**: 2-3 coordinated operations
- Basic implementations using existing patterns
- Simple modifications to existing code

**COMPLEX_PLAN**: 4+ comprehensive steps
- Complete system implementations
- Advanced features with extensive integration

**KEY**: Continuation requests referencing recent results = DIRECT (no new tools needed)

**USER NARRATION GUIDELINES:**
When providing the user_narration field, describe what you'll do to help them in 2-3 sentences:
- Focus on the OUTCOME and DETAILS of what you'll implement/fix/find
- Be specific about the features, steps, or solutions you'll provide
- Explain WHAT you'll do, not HOW the agent works internally
- Avoid phrases like: "using previously retrieved data", "no new tool calls needed", "filtering existing results"
- Instead describe: what you'll analyze, what features you'll add, what issues you'll fix

Examples of GOOD narrations (detailed, action-focused):
- "I'll help you identify which script handles connections by examining your networking code. I'll analyze the script structure to show you exactly where connection logic is implemented and how it's organized."
- "I'll create a comprehensive player movement script that includes WASD keyboard controls, smooth acceleration and deceleration, jumping with ground detection, and camera rotation. The script will be well-commented and ready to attach to your player GameObject."
- "I'll help you debug the connection issue in your networking code. I'll examine the current implementation, identify potential causes like timeout settings or serialization problems, and provide specific fixes to get your multiplayer working smoothly."
- "I'll analyze your project assets to find the largest one. I'll compare file sizes across all asset types including textures, models, audio files, and scenes to show you which one is taking up the most space and provide size details."

Examples of BAD narrations (too short or exposing internals):
- "I'll analyze those results for you. The request is a direct follow-up using previously retrieved data."
- "I can help with that right away."
- "I'll filter the existing results to show you the biggest one."
"""

    # Add explicit continuation context if detected
    if continuation_context:
        continuation_section = f"""

 CONTINUATION CONTEXT DETECTED 

The user just received these results from {continuation_context['tool_result']['tool_name']}:
**Result Summary:** {continuation_context['tool_result']['summary']}

**Previous interaction included:**
{json.dumps(continuation_context['tool_result']['result'], indent=2)[:500]}...

The current request "{user_request}" is a FOLLOW-UP asking for more analysis, filtering, or details about these existing results.

**CLASSIFICATION GUIDANCE FOR THIS CONTINUATION:**
- This should be classified as DIRECT (no new tools needed)
- The data already exists in memory
- Just needs to analyze/present existing results differently
- Direct_act can handle this with the existing context"""
        
        base_classification_prompt += continuation_section
    
    # Inject memory context
    print(f"\n [Classify] Injecting memory context into classification prompt...")
    enhanced_classification_prompt = await inject_memory_into_prompt(
        base_prompt=base_classification_prompt,
        state=state,
        include_patterns=True,
        include_episodes=False
    )
    
    # Log what was injected
    base_length = len(base_classification_prompt)
    enhanced_length = len(enhanced_classification_prompt)
    added_context = enhanced_length - base_length
    
    print(f"   Base prompt: {base_length} chars")
    print(f"   Enhanced prompt: {enhanced_length} chars")
    print(f"   Memory context added: {added_context} chars")
    
    # Create classification request with continuation awareness
    if continuation_context:
        classification_request = f""" CONTINUATION REQUEST DETECTED 

Current request: "{user_request}"
Previous tool: {continuation_context['tool_result']['tool_name']}
Previous result: {continuation_context['tool_result']['summary']}

This is clearly a FOLLOW-UP question about the previous results.

**Classify as DIRECT** because:
1. The data already exists in memory
2. No new tool execution is needed
3. Just needs to analyze/filter/present existing data
4. Direct_act can handle this using the continuation context

Provide your classification confirming this is a continuation request."""
    else:
        classification_request = f"""Classify this Unity/game development request: "{user_request}"
- Unity Version: {unity_version or "Unknown"}
- Project: {project_name or "Unknown"}

Analyze the request and determine:
1. How many production tools would be needed?
2. Does it need project analysis first?
3. Is this a single query/operation or coordinated tool usage?

Provide your complexity classification with reasoning."""
    
    print(f"\n [Classify] Sending to LLM:")
    print(f"   Model: {context.model}")
    print(f"   System prompt: {enhanced_length} chars")
    print(f"   User request: {len(classification_request)} chars")
    print(f"   Continuation detected: {continuation_context is not None}")
    
    # Structure messages
    messages = [
        {"role": "system", "content": enhanced_classification_prompt},
        {"role": "user", "content": classification_request}
    ]
    
    try:
        print(f"\n [Classify] Waiting for LLM classification...")
        
        # Get classification
        structured_model = model.with_structured_output(ComplexityAssessment)
        assessment = await structured_model.ainvoke(messages)
        
        print(f"\n [Classify] LLM Classification Received:")
        print(f"   Level: {assessment.complexity_level}")
        print(f"   Confidence: {assessment.confidence:.2f}")
        print(f"   Reasoning: {assessment.reasoning[:100]}...")
        print(f"   Suggested approach: {assessment.suggested_approach[:100]}...")
        
        # Use the LLM-generated user narration
        user_narration = assessment.user_narration
        
        # Store metadata
        updated_metadata = {
            **state.runtime_metadata,
            "complexity_level": assessment.complexity_level,
            "complexity_reasoning": assessment.reasoning,
            "complexity_confidence": assessment.confidence,
            "suggested_approach": assessment.suggested_approach,
            "is_continuation": continuation_context is not None,
            "project_id": project_id,
            "project_root": project_root,
            "project_name": project_name,
            "unity_version": unity_version,
            "sqlite_path": sqlite_path,
            "config_received_at": submitted_at,
        }
        
        logger.info(f"📊 [Classify] Classified as: {assessment.complexity_level}")
        
        # Update memory focus
        if memory_to_use and user_request:
            entities = extract_entities_from_request(user_request)
            topics = extract_topics_from_request(user_request)
            await memory_to_use.update_focus(entities, topics)
            memory_to_use.add_message("user", user_request)
        
        return {
            "runtime_metadata": updated_metadata,
            "messages": [AIMessage(content=user_narration)],
            **memory_updates
        }
        
    except Exception as e:
        logger.error(f"❌ [Classify] Classification error: {e}")
        
        # Enhanced fallback with continuation awareness
        if continuation_context:
            fallback_level = "direct"
            fallback_reason = "Continuation request detected - using existing context"
        else:
            request_words = len(user_request.split())
            if request_words <= 8:
                fallback_level = "direct"
                fallback_reason = "Short query - direct response"
            elif request_words <= 15:
                fallback_level = "simple_plan"
                fallback_reason = "Moderate request - simple planning"
            else:
                fallback_level = "complex_plan"
                fallback_reason = "Complex request - full planning"
        
        # Update memory even in fallback
        if memory_to_use and user_request:
            entities = extract_entities_from_request(user_request)
            topics = extract_topics_from_request(user_request)
            await memory_to_use.update_focus(entities, topics)
            memory_to_use.add_message("user", user_request)
        
        return {
            "runtime_metadata": {
                **state.runtime_metadata,
                "complexity_level": fallback_level,
                "complexity_reasoning": fallback_reason,
                "complexity_confidence": 0.5,
                "is_continuation": continuation_context is not None,
                "classification_error": str(e),
                "project_id": project_id,
                "project_root": project_root,
                "project_name": project_name,
                "unity_version": unity_version,
                "sqlite_path": sqlite_path,
            },
            "messages": [AIMessage(content=f"I'll help you with: {user_request}")],
            **memory_updates
        }