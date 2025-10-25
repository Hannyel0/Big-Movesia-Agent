"""UIMessage emission utilities for LangGraph SDK."""

import uuid
from typing import Any, Dict

from langchain_core.messages import AIMessage
from react_agent.state import ExecutionPlan


def create_web_search_ui_message(
    search_result: Dict[str, Any], message_id: str
) -> AIMessage:
    """Create a UIMessage-compatible AIMessage for web search results.

    Args:
        search_result: The search result from web_search tool
        message_id: The ID of the AI message this UI should attach to

    Returns:
        AIMessage formatted as a UIMessage for the frontend
    """
    ui_message_id = str(uuid.uuid4())

    return AIMessage(
        content="",  # Empty content for UI-only message
        id=ui_message_id,
        additional_kwargs={
            "ui_message": True,
            "ui_type": "web_search_results",
            "ui_data": search_result,
            "message_id": message_id,  # Link to parent AI message
        },
    )


def create_plan_ui_message(plan: ExecutionPlan, message_id: str) -> AIMessage:
    """Create a UIMessage-compatible AIMessage for plan visualization.

    Args:
        plan: The ExecutionPlan object
        message_id: The ID of the AI message this UI should attach to

    Returns:
        AIMessage formatted as a UIMessage for the frontend
    """
    ui_message_id = str(uuid.uuid4())

    # Convert plan to serializable format
    plan_data = {
        "goal": plan.goal,
        "steps": [
            {
                "description": step.description,
                "tool_name": step.tool_name,
                "success_criteria": step.success_criteria,
                "dependencies": step.dependencies,
                "status": step.status,
            }
            for step in plan.steps
        ],
        "metadata": plan.metadata,
        "total_steps": len(plan.steps),
    }

    return AIMessage(
        content="",  # Empty content for UI-only message
        id=ui_message_id,
        additional_kwargs={
            "ui_message": True,
            "ui_type": "execution_plan",
            "ui_data": plan_data,
            "message_id": message_id,  # Link to parent AI message
        },
    )
