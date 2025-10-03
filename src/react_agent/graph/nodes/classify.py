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
                "project_name": project_name,
                "unity_version": unity_version,
                "sqlite_path": sqlite_path,
            }
        }
    
    # Static classification prompt - optimized for caching
    static_classification_content = """You are a Unity/Unreal Engine development complexity assessor. Classify user requests into three categories:

**DIRECT**: Simple queries that can be answered immediately without planning or multiple tools
- Information requests ("what is", "how do I", "explain", "show me")
- Single tool operations ("get project info", "compile my project", "search for X")
- Quick status checks or basic questions
Examples: "What's in my project?", "Compile my code", "How do character controllers work?"

**SIMPLE_PLAN**: Straightforward development tasks requiring 2-3 coordinated steps
- Basic asset creation with minimal complexity
- Simple script implementations following standard patterns
- Configuration changes with verification
- Small debugging or fixing tasks
Examples: "Create a basic movement script", "Add a health bar to my UI", "Fix compilation errors"

**COMPLEX_PLAN**: Multi-faceted development requiring comprehensive planning (4+ steps)
- Complete system implementations
- Advanced features requiring multiple integrations
- Complex troubleshooting across multiple components
- Architecture-level changes or major feature additions
Examples: "Build a complete inventory system", "Create an AI enemy with pathfinding", "Implement multiplayer networking"

Classification Guidelines:
- Consider the scope and depth of work required
- Evaluate how many different tools/steps would be needed
- Assess whether the task requires research, implementation, and testing phases
- Factor in complexity of integration with existing systems
- Consider if the request requires domain expertise or just standard operations

Provide clear reasoning for your classification to help the system choose the optimal execution path."""

    # Dynamic request content with project context awareness
    classification_request = f"""Classify this Unity/game development request: "{user_request}"

Project Context:
- Unity Version: {unity_version or "Unknown"}
- Project: {project_name or "Unknown"}

Analyze the request and determine:
1. How many distinct development steps would be required?
2. Does it need research, or is the approach straightforward?
3. How much coordination between different tools/systems is needed?
4. Is this a standard pattern or something requiring custom implementation?

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