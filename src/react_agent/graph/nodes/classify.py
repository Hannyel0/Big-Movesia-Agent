"""Complexity classification node for the ReAct agent with runtime config support."""

from __future__ import annotations

from typing import Any, Dict, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from react_agent.context import Context
from react_agent.state import State
from react_agent.utils import get_message_text, get_model


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
    """Classify request complexity using LLM to determine execution strategy.
    
    Now reads runtime project context from config.configurable and stores it
    in runtime_metadata for downstream nodes to access.
    """
    
    # ✅ NEW: Extract project context from config.configurable
    configurable = config.get("configurable", {})
    project_id = configurable.get("project_id", "")
    project_root = configurable.get("project_root", "")
    project_name = configurable.get("project_name", "")
    unity_version = configurable.get("unity_version", "")
    sqlite_path = configurable.get("sqlite_path", "")
    submitted_at = configurable.get("submitted_at", "")
    
    # ✅ NEW: Log project context for debugging
    print(f"\n{'='*60}")
    print(f"🔧 [Classify] Runtime Config Received:")
    print(f"  Project ID: {project_id}")
    print(f"  Project Root: {project_root}")
    print(f"  Project Name: {project_name}")
    print(f"  Unity Version: {unity_version}")
    print(f"  SQLite Path: {sqlite_path}")
    print(f"  Submitted At: {submitted_at}")
    print(f"{'='*60}\n")
    
    # ✅ NEW: Validate that we received project context
    if not project_id or not project_root:
        print(f"⚠️  [Classify] WARNING: Missing critical project context!")
        print(f"   - project_id: {project_id or 'MISSING'}")
        print(f"   - project_root: {project_root or 'MISSING'}")
    
    context = runtime.context
    model = get_model(context.model)
    
    # Extract user request from messages
    user_request = None
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            user_request = get_message_text(msg)
            break
    
    if not user_request:
        # Fallback to complex planning if no clear request
        return {
            "runtime_metadata": {
                **state.runtime_metadata,
                "complexity_level": "complex_plan",
                "complexity_reasoning": "No clear request found - defaulting to full planning",
                # ✅ NEW: Store project context in runtime_metadata
                "project_id": project_id,
                "project_root": project_root,
                "unity_version": unity_version,
                "sqlite_path": sqlite_path,
            }
        }
    
    # Static classification prompt - optimized for caching
    static_classification_content = """You are a Unity/Unreal Engine development complexity assessor with access to production tools. Classify user requests into three categories:

**Available Production Tools:**
- **search_project**: Query indexed project database (assets, hierarchy, components, dependencies)
- **code_snippets**: Semantic search through C# scripts by functionality
- **file_operation**: Safe file I/O with validation (read/write/modify/delete/move/diff)
- **web_search**: Research Unity documentation, tutorials, best practices

**DIRECT**: Simple queries requesting data from the user's project
- Queries for existing assets, GameObjects, components, scripts IN YOUR PROJECT
- Single database queries or file reads FROM YOUR PROJECT
- Examples: "What assets do I have?", "Show me my GameObjects", "List my scripts", "What's in my project?"
- KEY: If asking for data that EXISTS in the project → DIRECT + TOOL_CALL

**NOT DIRECT** (should provide guidance):
- Conceptual questions: "What IS a prefab?", "How DO character controllers work?"
- Tutorial requests: "Teach me about physics", "Explain the Input System"
- These get informational responses, not tool execution

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
- CRITICAL: Distinguish between "show me MY data" (DIRECT) vs "explain a concept" (informational)
- Consider how many production tools would be needed
- Evaluate if the task requires project analysis before implementation
- Assess whether existing code needs to be found and understood first
- Factor in complexity of file operations and integrations
- Consider if the request needs research + analysis + implementation phases

Provide clear reasoning for your classification to help the system choose the optimal execution path."""

    # Dynamic request content with project context awareness
    classification_request = f"""Classify this Unity/game development request: "{user_request}"
- Unity Version: {unity_version or "Unknown"}
- Project: {project_name or "Unknown"}

Analyze the request and determine:
1. How many production tools would be needed (search_project, code_snippets, file_operation, web_search)?
2. Does it need project analysis or existing code discovery first?
3. Is this a single query/operation or does it require coordinated tool usage?
4. Does it need research, code analysis, and implementation phases?

Provide your complexity classification with reasoning."""

    # Structure messages for optimal caching
    messages = [
        {"role": "system", "content": static_classification_content},
        {"role": "user", "content": classification_request}
    ]
    
    try:
        # Use structured output for reliable classification
        structured_model = model.with_structured_output(ComplexityAssessment)
        assessment = await structured_model.ainvoke(messages)
        
        # Create user-friendly narration based on classification
        narration_map = {
            "direct": f"I can help you with that right away. {assessment.reasoning}",
            "simple_plan": f"I'll handle this with a focused approach. {assessment.reasoning}",
            "complex_plan": f"This is a comprehensive request that I'll break down systematically. {assessment.reasoning}"
        }
        
        user_narration = narration_map.get(
            assessment.complexity_level,
            "Let me analyze this request and create an appropriate plan."
        )
        
        # ✅ NEW: Store classification AND project context in runtime metadata
        updated_metadata = {
            **state.runtime_metadata,
            "complexity_level": assessment.complexity_level,
            "complexity_reasoning": assessment.reasoning,
            "complexity_confidence": assessment.confidence,
            "suggested_approach": assessment.suggested_approach,
            # ✅ NEW: Propagate project context to all downstream nodes
            "project_id": project_id,
            "project_root": project_root,
            "project_name": project_name,
            "unity_version": unity_version,
            "sqlite_path": sqlite_path,
            "config_received_at": submitted_at,
        }
        
        print(f"✅ [Classify] Classification complete: {assessment.complexity_level}")
        print(f"   Confidence: {assessment.confidence:.2f}")
        print(f"   Project context stored in runtime_metadata for downstream nodes\n")
        
        return {
            "runtime_metadata": updated_metadata,
            "messages": [AIMessage(content=user_narration)]
        }
        
    except Exception as e:
        # Fallback classification with error handling
        print(f"⚠️  [Classify] Error during classification: {e}")
        
        # Analyze request length and keywords as backup
        request_words = len(user_request.split())
        
        if request_words <= 8 and any(starter in user_request.lower() for starter in 
                                    ["what", "how", "where", "when", "explain", "show", "tell"]):
            fallback_level = "direct"
            fallback_reason = "Short informational query - direct response appropriate"
        elif request_words <= 15 and any(simple_word in user_request.lower() for simple_word in 
                                       ["basic", "simple", "quick", "add", "create a", "make a"]):
            fallback_level = "simple_plan"
            fallback_reason = "Moderate request scope - simple planning approach"
        else:
            fallback_level = "complex_plan"
            fallback_reason = "Comprehensive or unclear request - full planning recommended"
        
        fallback_narration = f"I'll help you with this request using a {fallback_level.replace('_', ' ')} approach."
        
        return {
            "runtime_metadata": {
                **state.runtime_metadata,
                "complexity_level": fallback_level,
                "complexity_reasoning": fallback_reason,
                "complexity_confidence": 0.5,
                "classification_error": str(e),
                # ✅ NEW: Include project context even in fallback
                "project_id": project_id,
                "project_root": project_root,
                "project_name": project_name,
                "unity_version": unity_version,
                "sqlite_path": sqlite_path,
            },
            "messages": [AIMessage(content=fallback_narration)]
        }