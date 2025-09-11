"""Finish node for the ReAct agent."""

from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.state import State
from react_agent.narration import NarrationEngine, StreamingNarrator


# Initialize narration components
narration_engine = NarrationEngine()


async def finish(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Enhanced finish node with rich completion summary."""
    context = runtime.context
    
    # Use narration engine for rich completion summary
    completion_narration = narration_engine.create_completion_narration(
        state.completed_steps,
        state.plan
    )
    
    # Add final streaming updates if supported
    if context.runtime_metadata.get("supports_streaming"):
        streaming_narrator = StreamingNarrator()
        final_update = streaming_narrator.create_inline_update(
            "✨ Implementation complete! Check your Unity project for the new features.",
            style="success"
        )
        if hasattr(runtime, 'push_ui_message'):
            runtime.push_ui_message(final_update)
    
    return {
        "messages": [AIMessage(content=completion_narration)]
    }