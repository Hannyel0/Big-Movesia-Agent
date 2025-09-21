"""Micro-retry node for handling transient errors without planning overhead."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.state import State

# Set up logging
logger = logging.getLogger(__name__)

# Micro-retry configuration
MICRO_RETRY_CONFIG = {
    "base_delay": 0.5,  # Base delay in seconds
    "max_delay": 2.0,   # Maximum delay in seconds
    "backoff_multiplier": 1.5,  # Exponential backoff multiplier
    "max_micro_retries": 2,  # Maximum micro-retries before escalating
}


async def micro_retry(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """
    Handle micro-retry for transient errors with exponential backoff.
    
    This node provides immediate retry capability for transient errors like
    network timeouts, rate limits, or temporary service unavailability without
    the overhead of creating recovery plans.
    """
    logger.info("=== EXECUTING MICRO-RETRY ===")
    
    # Calculate delay based on retry count
    retry_count = state.retry_count
    base_delay = MICRO_RETRY_CONFIG["base_delay"]
    multiplier = MICRO_RETRY_CONFIG["backoff_multiplier"]
    max_delay = MICRO_RETRY_CONFIG["max_delay"]
    
    # Exponential backoff with jitter
    delay = min(base_delay * (multiplier ** retry_count), max_delay)
    
    logger.info(f"Micro-retry #{retry_count + 1} with {delay:.1f}s delay")
    
    # Apply delay for transient error recovery
    await asyncio.sleep(delay)
    
    # Create contextual retry message
    retry_message = _create_micro_retry_message(state, retry_count + 1)
    
    # Check if we should escalate to full error recovery
    should_escalate = (retry_count + 1) >= MICRO_RETRY_CONFIG["max_micro_retries"]
    
    if should_escalate:
        logger.warning(f"Micro-retry limit reached ({MICRO_RETRY_CONFIG['max_micro_retries']}), escalating to error recovery")
        return {
            "retry_count": retry_count + 1,
            "current_assessment": None,
            "should_micro_retry": False,
            "needs_error_recovery": True,  # Escalate to full recovery
            "messages": [AIMessage(content=f"{retry_message} If this continues, I'll use a more comprehensive fix.")],
            "runtime_metadata": {
                **state.runtime_metadata,
                "micro_retry_escalated": True,
                "micro_retry_attempts": retry_count + 1
            }
        }
    
    # Continue with micro-retry
    return {
        "retry_count": retry_count + 1,
        "current_assessment": None,
        "should_micro_retry": False,  # Clear the flag so we proceed to act
        "messages": [AIMessage(content=retry_message)],
        "runtime_metadata": {
            **state.runtime_metadata,
            "micro_retry_attempted": True,
            "micro_retry_count": retry_count + 1,
            "micro_retry_delay": delay
        }
    }


def _create_micro_retry_message(state: State, attempt_num: int) -> str:
    """Create contextual micro-retry message based on current step."""
    if not state.plan or state.step_index >= len(state.plan.steps):
        return f"Retrying due to transient issue (attempt {attempt_num})..."
    
    current_step = state.plan.steps[state.step_index]
    tool_name = current_step.tool_name
    
    # Tool-specific retry messages
    tool_messages = {
        "search": "Reconnecting to search services",
        "get_project_info": "Refreshing project connection", 
        "compile_and_test": "Retrying build process",
        "write_file": "Attempting file write again",
        "create_asset": "Retrying asset creation",
        "get_script_snippets": "Reconnecting to code repository",
        "scene_management": "Refreshing scene access",
        "edit_project_config": "Retrying configuration update"
    }
    
    base_message = tool_messages.get(tool_name, f"Retrying {tool_name}")
    
    # Add attempt context
    if attempt_num == 1:
        return f"{base_message} due to transient issue..."
    elif attempt_num == 2:
        return f"{base_message} again with adjusted parameters..."
    else:
        return f"{base_message} (attempt {attempt_num})..."


def should_micro_retry(error_category: str, error_message: str, retry_count: int) -> bool:
    """
    Determine if an error should trigger micro-retry instead of normal retry.
    
    Args:
        error_category: Category of the error
        error_message: The actual error message
        retry_count: Current retry count for this step
    
    Returns:
        True if micro-retry should be attempted
    """
    # Don't micro-retry if we've already tried multiple times
    if retry_count >= MICRO_RETRY_CONFIG["max_micro_retries"]:
        return False
    
    # Check for transient error categories
    transient_categories = [
        "network_error",
        "tool_malfunction", 
        "timeout",
        "rate_limit",
        "service_unavailable"
    ]
    
    if error_category.lower() in transient_categories:
        return True
    
    # Check for transient error patterns in message
    error_lower = error_message.lower()
    transient_patterns = [
        "network", "timeout", "connection", "rate limit",
        "temporary", "try again", "service unavailable",
        "internal server error", "502", "503", "504",
        "connection reset", "connection refused"
    ]
    
    return any(pattern in error_lower for pattern in transient_patterns)


def get_micro_retry_stats(state: State) -> Dict[str, Any]:
    """Get micro-retry statistics for debugging and monitoring."""
    metadata = state.runtime_metadata
    
    return {
        "micro_retry_attempted": metadata.get("micro_retry_attempted", False),
        "micro_retry_count": metadata.get("micro_retry_count", 0),
        "micro_retry_escalated": metadata.get("micro_retry_escalated", False),
        "last_micro_retry_delay": metadata.get("micro_retry_delay", 0),
        "max_micro_retries": MICRO_RETRY_CONFIG["max_micro_retries"],
        "current_retry_count": state.retry_count
    }