"""Complexity classification node for the ReAct agent with runtime config support."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from react_agent.context import Context
from react_agent.state import State
from react_agent.utils import get_message_text, get_model
from react_agent.memory import (
    initialize_memory_system, 
    extract_entities_from_request, 
    extract_topics_from_request,
    inject_memory_into_prompt
)



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
    confidence: float = Field(
        default=0.8, 
        ge=0.0, 
        le=1.0, 
        description="Confidence in the complexity assessment"
    )


async def classify(
    state: State, 
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> Dict[str, Any]:
    """Classify request complexity using LLM with memory context."""
    
    print(f"\n{'='*60}")
    print(f"🎯 [CLASSIFY] ===== STARTING CLASSIFICATION =====")
    print(f"{'='*60}")
    
    # Memory initialization
    memory_updates = {}
    if not state.memory:
        print(f"🧠 [Classify] No memory found - initializing...")
        memory_updates = await initialize_memory_system(state, config)
        if memory_updates.get("memory"):
            print(f"✅ [Classify] Memory system initialized successfully")
    else:
        print(f"✅ [Classify] Memory system already initialized")
    
    # Extract project context
    configurable = config.get("configurable", {})
    project_id = configurable.get("project_id", "")
    project_root = configurable.get("project_root", "")
    project_name = configurable.get("project_name", "")
    unity_version = configurable.get("unity_version", "")
    sqlite_path = configurable.get("sqlite_path", "")
    submitted_at = configurable.get("submitted_at", "")
    
    print(f"\n📦 [Classify] Project Context:")
    print(f"   Project: {project_name} (ID: {project_id})")
    print(f"   Unity: {unity_version}")
    print(f"   Root: {project_root}")
    
    if not project_id or not project_root:
        print(f"⚠️  [Classify] WARNING: Missing critical project context!")
    
    context = runtime.context
    model = get_model(context.model)
    
    # Extract user request
    user_request = None
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            user_request = get_message_text(msg)
            break
    
    if not user_request:
        print(f"❌ [Classify] No user request found - using fallback")
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
    
    print(f"\n💬 [Classify] User Request:")
    print(f"   \"{user_request}\"")
    print(f"   Length: {len(user_request)} chars, {len(user_request.split())} words")
    
    # Get memory reference
    memory_to_use = memory_updates.get("memory") or state.memory
    
    # ✅ NEW: Log memory state BEFORE injection
    if memory_to_use:
        print(f"\n🧠 [Classify] Memory State Before Classification:")
        if hasattr(memory_to_use, 'working_memory'):
            wm = memory_to_use.working_memory
            tool_count = len(wm.recent_tool_results)
            print(f"   Recent tool results: {tool_count}")
            
            if tool_count > 0:
                for i, tool in enumerate(wm.recent_tool_results[-3:], 1):
                    print(f"   {i}. {tool['tool_name']}: {tool['summary'][:60]}")
            
            print(f"   Focus entities: {wm.focus_entities[:3] if wm.focus_entities else 'None'}")
            print(f"   Focus topics: {wm.focus_topics[:3] if wm.focus_topics else 'None'}")
        else:
            print(f"   Working memory not available")
    else:
        print(f"\n⚠️  [Classify] No memory available")
    
    # ✅ CRITICAL: For ALL classifications, enhance the prompt with memory context
    # This way the LLM can see recent interactions even for new requests
    base_classification_prompt = """You are a Unity/Unreal Engine development complexity assessor with access to production tools. Classify user requests into three categories:

**Available Production Tools:**
- **search_project**: Query indexed project database (assets, hierarchy, components, dependencies)
- **code_snippets**: Semantic search through C# scripts by functionality
- **file_operation**: Safe file I/O with validation (read/write/modify/delete/move/diff)
- **web_search**: Research Unity documentation, tutorials, best practices

**DIRECT**: Single-step responses (can answer immediately)
- Project data queries: "What assets do I have?", "Show me my GameObjects", "List my scripts"
  → These execute a single tool (search_project, code_snippets, etc.)
  
- Informational queries: "What is a prefab?", "How do character controllers work?", "Explain physics"
  → These provide knowledge-based answers without executing tools
  
- KEY: If it can be answered in ONE step (tool call OR explanation) → DIRECT

**SIMPLE_PLAN**: Straightforward tasks requiring 2-3 coordinated tool operations
- Basic implementations using existing patterns
- Simple modifications to existing code
- Creating new files based on found examples
- Small debugging using project queries and code search
Examples: "Create a basic movement script", "Add a health bar to my UI", "Fix my player controller"

**COMPLEX_PLAN**: Multi-faceted development requiring comprehensive tool coordination (4+ steps)
- Complete system implementations requiring research, code analysis, and multiple file operations
- Advanced features needing extensive project integration
- Complex troubleshooting across multiple components and files
- Architecture-level changes requiring thorough project understanding
Examples: "Build a complete inventory system", "Create an AI enemy with pathfinding", "Implement multiplayer networking"

Classification Guidelines:
- Count the number of distinct steps/tools needed
- DIRECT = 1 step (whether it's a tool call or an explanation)
- SIMPLE_PLAN = 2-3 coordinated steps
- COMPLEX_PLAN = 4+ coordinated steps
- Consider if the task requires project analysis before implementation
- Assess whether existing code needs to be found and understood first
- Factor in complexity of file operations and integrations

Provide clear reasoning for your classification to help the system choose the optimal execution path."""

    # ✅ NEW: Log memory context injection
    print(f"\n🔄 [Classify] Injecting memory context into classification prompt...")
    enhanced_classification_prompt = await inject_memory_into_prompt(
        base_prompt=base_classification_prompt,
        state=state,
        include_patterns=True,
        include_episodes=False
    )
    
    # ✅ NEW: Log what was injected
    base_length = len(base_classification_prompt)
    enhanced_length = len(enhanced_classification_prompt)
    added_context = enhanced_length - base_length
    
    print(f"   Base prompt: {base_length} chars")
    print(f"   Enhanced prompt: {enhanced_length} chars")
    print(f"   Memory context added: {added_context} chars")
    
    if added_context > 0:
        print(f"\n📝 [Classify] Memory Context Preview (first 300 chars):")
        # Extract just the memory context part
        memory_section = enhanced_classification_prompt[base_length:base_length+300]
        print(f"   {memory_section}...")
    else:
        print(f"   ⚠️  No memory context was added")
    
    # Create classification request
    classification_request = f"""Classify this Unity/game development request: "{user_request}"
- Unity Version: {unity_version or "Unknown"}
- Project: {project_name or "Unknown"}

Analyze the request and determine:
1. How many production tools would be needed?
2. Does it need project analysis first?
3. Is this a single query/operation or coordinated tool usage?

Provide your complexity classification with reasoning."""
    
    print(f"\n📤 [Classify] Sending to LLM:")
    print(f"   Model: {context.model}")
    print(f"   System prompt: {enhanced_length} chars")
    print(f"   User request: {len(classification_request)} chars")
    
    # Structure messages
    messages = [
        {"role": "system", "content": enhanced_classification_prompt},
        {"role": "user", "content": classification_request}
    ]
    
    try:
        print(f"\n⏳ [Classify] Waiting for LLM classification...")
        
        # Get classification
        structured_model = model.with_structured_output(ComplexityAssessment)
        assessment = await structured_model.ainvoke(messages)
        
        print(f"\n✅ [Classify] LLM Classification Received:")
        print(f"   Level: {assessment.complexity_level}")
        print(f"   Confidence: {assessment.confidence:.2f}")
        print(f"   Reasoning: {assessment.reasoning[:100]}...")
        print(f"   Suggested approach: {assessment.suggested_approach[:100]}...")
        
        # Create narration
        narration_map = {
            "direct": f"I can help you with that right away. {assessment.reasoning}",
            "simple_plan": f"I'll handle this with a focused approach. {assessment.reasoning}",
            "complex_plan": f"This is a comprehensive request that I'll break down systematically. {assessment.reasoning}"
        }
        
        user_narration = narration_map.get(
            assessment.complexity_level,
            "Let me analyze this request and create an appropriate plan."
        )
        
        # Store metadata
        updated_metadata = {
            **state.runtime_metadata,
            "complexity_level": assessment.complexity_level,
            "complexity_reasoning": assessment.reasoning,
            "complexity_confidence": assessment.confidence,
            "suggested_approach": assessment.suggested_approach,
            "project_id": project_id,
            "project_root": project_root,
            "project_name": project_name,
            "unity_version": unity_version,
            "sqlite_path": sqlite_path,
            "config_received_at": submitted_at,
        }
        
        print(f"\n📊 [Classify] Routing Decision:")
        print(f"   Will route to: {assessment.complexity_level}")
        
        # Update memory focus
        if memory_to_use and user_request:
            print(f"\n🧠 [Classify] Updating memory focus...")
            entities = extract_entities_from_request(user_request)
            topics = extract_topics_from_request(user_request)
            
            print(f"   Extracted entities: {entities[:5] if entities else 'None'}")
            print(f"   Extracted topics: {topics[:5] if topics else 'None'}")
            
            await memory_to_use.update_focus(entities, topics)
            memory_to_use.add_message("user", user_request)
            
            print(f"✅ [Classify] Memory focus updated")
        
        print(f"\n{'='*60}")
        print(f"🎯 [CLASSIFY] ===== CLASSIFICATION COMPLETE =====")
        print(f"{'='*60}\n")
        
        return {
            "runtime_metadata": updated_metadata,
            "messages": [AIMessage(content=user_narration)],
            **memory_updates
        }
        
    except Exception as e:
        print(f"\n❌ [Classify] ERROR during classification:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        print(f"   Using fallback classification...")
        
        # Fallback logic
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
        
        print(f"   Fallback level: {fallback_level}")
        print(f"   Fallback reason: {fallback_reason}")
        
        # Update memory even in fallback
        if memory_to_use and user_request:
            entities = extract_entities_from_request(user_request)
            topics = extract_topics_from_request(user_request)
            await memory_to_use.update_focus(entities, topics)
            memory_to_use.add_message("user", user_request)
            print(f"✅ [Classify] Memory updated despite error")
        
        print(f"\n{'='*60}")
        print(f"🎯 [CLASSIFY] ===== CLASSIFICATION COMPLETE (FALLBACK) =====")
        print(f"{'='*60}\n")
        
        return {
            "runtime_metadata": {
                **state.runtime_metadata,
                "complexity_level": fallback_level,
                "complexity_reasoning": fallback_reason,
                "complexity_confidence": 0.5,
                "classification_error": str(e),
                "project_id": project_id,
                "project_root": project_root,
                "project_name": project_name,
                "unity_version": unity_version,
                "sqlite_path": sqlite_path,
            },
            "messages": [AIMessage(content=f"I'll help with a {fallback_level.replace('_', ' ')} approach.")],
            **memory_updates
        }