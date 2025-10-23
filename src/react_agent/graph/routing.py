"""Enhanced routing functions with micro-retry support and completion detection."""

from __future__ import annotations
from typing import Literal
import json
import logging
from langchain_core.messages import AIMessage, ToolMessage
from react_agent.state import State
from react_agent.memory import get_memory_insights

logger = logging.getLogger(__name__)


def route_after_assess(state: State) -> Literal["advance_step", "increment_retry", "micro_retry", "error_recovery", "finish"]:
    """Enhanced routing with micro-retry support and absolute completion priority."""
    
    # Safety checks
    if not state.current_assessment:
        logger.info("🔀 [Router] ASSESS → FINISH (no assessment)")
        return "finish"
    
    if not state.plan or not state.plan.steps:
        logger.info("🔀 [Router] ASSESS → FINISH (no plan)")
        return "finish"
    
    # NEW: Check for micro-retry flag first (highest priority for transient errors)
    if getattr(state, "should_micro_retry", False):
        logger.info("🔀 [Router] ASSESS → MICRO_RETRY (transient error)")
        return "micro_retry"
    
    # ABSOLUTE PRIORITY: Check if we're on or past the last step
    last_step_index = len(state.plan.steps) - 1
    is_on_or_past_last_step = state.step_index >= last_step_index
    
    # If we're on the last step AND it succeeded, ALWAYS finish
    if is_on_or_past_last_step and state.current_assessment.outcome == "success":
        logger.info(f"🔀 [Router] ASSESS → FINISH (last step {state.step_index + 1}/{len(state.plan.steps)} success)")
        return "finish"
    
    # If we somehow went past the last step, force finish
    if state.step_index >= len(state.plan.steps):
        logger.info(f"🔀 [Router] ASSESS → FINISH (step {state.step_index + 1} exceeds plan)")
        return "finish"
    
    # Check for error recovery need (only if not completing)
    if getattr(state, "needs_error_recovery", False) or state.runtime_metadata.get("needs_error_recovery"):
        logger.info("🔀 [Router] ASSESS → ERROR_RECOVERY")
        return "error_recovery"
    
    # Handle success on non-final steps
    if state.current_assessment.outcome == "success":
        logger.info(f"🔀 [Router] ASSESS → ADVANCE_STEP (step {state.step_index + 1} success)")
        return "advance_step"
    
    # Handle retry logic
    if state.current_assessment.outcome == "retry":
        # ✅ OPTIMIZATION: Check memory insights before retry
        if state.memory:
            insights = get_memory_insights(state)
            relevant_patterns = insights.get("relevant_patterns", [])
            
            # If this pattern usually fails, skip retry and go to error recovery
            for pattern in relevant_patterns:
                success_rate = pattern.get("success_rate", 1.0)
                if success_rate < 0.3:  # Less than 30% success rate
                    logger.info(f"🔀 [Router] ASSESS → ERROR_RECOVERY (low success rate {success_rate:.0%})")
                    return "error_recovery"
        
        if state.retry_count >= state.max_retries_per_step:
            logger.info(f"🔀 [Router] ASSESS → ERROR_RECOVERY (max retries {state.retry_count})")
            return "error_recovery"
        
        logger.info(f"🔀 [Router] ASSESS → INCREMENT_RETRY (attempt {state.retry_count + 1})")
        return "increment_retry"
    
    # Handle blocked steps
    if state.current_assessment.outcome == "blocked":
        logger.info("🔀 [Router] ASSESS → ERROR_RECOVERY (blocked)")
        return "error_recovery"
    
    # Fallback to finish
    logger.info(f"🔀 [Router] ASSESS → FINISH (fallback: {state.current_assessment.outcome})")
    return "finish"


def should_continue(state: State) -> Literal["classify", "act"]:
    """Route to classify if no plan exists, otherwise act."""
    if state.plan is None:
        logger.info("🔀 [Router] START → CLASSIFY (no plan)")
        return "classify"
    logger.info("🔀 [Router] START → ACT (plan exists)")
    return "act"


def route_after_classify(state: State) -> Literal["direct_act", "simple_plan", "plan"]:
    """Route based on complexity classification."""
    complexity_level = state.runtime_metadata.get("complexity_level", "complex_plan")
    
    if complexity_level == "direct":
        logger.info("🔀 [Router] CLASSIFY → DIRECT_ACT (complexity=direct)")
        return "direct_act"
    elif complexity_level == "simple_plan":
        logger.info("🔀 [Router] CLASSIFY → SIMPLE_PLAN (complexity=simple_plan)")
        return "simple_plan"
    else:
        logger.info(f"🔀 [Router] CLASSIFY → PLAN (complexity={complexity_level})")
        return "plan"


def route_after_plan(state: State) -> Literal["act"]:
    """After planning, always proceed to action."""
    logger.info("🔀 [Router] PLAN → ACT")
    return "act"


def route_after_simple_plan(state: State) -> Literal["act"]:
    """After simple planning, proceed to action."""
    logger.info("🔀 [Router] SIMPLE_PLAN → ACT")
    return "act"


def route_after_direct_act(state: State) -> Literal["tools", "__end__"]:
    """Route after direct action - to tools if tool calls exist, otherwise end."""
    if state.messages and isinstance(state.messages[-1], AIMessage):
        if state.messages[-1].tool_calls:
            logger.info("🔀 [Router] DIRECT_ACT → TOOLS (has tool calls)")
            return "tools"
    logger.info("🔀 [Router] DIRECT_ACT → END (no tool calls)")
    return "__end__"


def route_after_act(state: State) -> Literal["tools", "assess"]:
    """Route after action - to tools if tool calls exist, otherwise assess."""
    if state.messages and isinstance(state.messages[-1], AIMessage):
        if state.messages[-1].tool_calls:
            logger.info("🔀 [Router] ACT → TOOLS (has tool calls)")
            return "tools"
    logger.info("🔀 [Router] ACT → ASSESS (no tool calls)")
    return "assess"


def route_after_error_recovery(state: State) -> Literal["act", "finish"]:
    """Route after error recovery execution."""
    if state.plan and state.step_index < len(state.plan.steps):
        logger.info(f"🔀 [Router] ERROR_RECOVERY → ACT (step {state.step_index + 1}/{len(state.plan.steps)})")
        return "act"
    logger.info("🔀 [Router] ERROR_RECOVERY → FINISH")
    return "finish"


def route_after_micro_retry(state: State) -> Literal["act"]:
    """Route after micro-retry - always back to act to retry the tool."""
    logger.info("🔀 [Router] MICRO_RETRY → ACT (retrying tool)")
    return "act"


def route_after_tools(state: State) -> Literal["check_file_approval", "assess"]:
    """
    Route after tool execution to check if approval is needed.
    Also captures tool results in memory for context.
    """
    import time
    routing_start = time.perf_counter()
    logger.info(f"⏱️  [ROUTER] route_after_tools() started")

    # ✅ CRITICAL: Capture tool result in memory FIRST
    extract_start = time.perf_counter()
    last_tool_message = None
    for msg in reversed(state.messages):
        if isinstance(msg, ToolMessage):
            last_tool_message = msg
            break

    if not last_tool_message:
        return "assess"

    extract_duration = (time.perf_counter() - extract_start) * 1000
    logger.info(f"⏱️  [ROUTER]   Extract tool message: {extract_duration:.1f}ms")

    # ✅ CRITICAL: Store tool result in memory
    if state.memory:
        try:
            memory_start = time.perf_counter()
            result = json.loads(last_tool_message.content)
            tool_name = last_tool_message.name or "unknown"
            logger.info(f"⏱️  [ROUTER]   Storing {tool_name} result in memory...")
            
            # Extract query from the preceding AIMessage
            query = ""
            args = {}
            for msg in reversed(state.messages):
                if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        if tc.get("name") == tool_name:
                            args = tc.get("args", {})
                            
                            # ✅ FIX: Check for query_description specifically for search_project
                            if tool_name == "search_project":
                                query = args.get("query_description", "")
                            else:
                                query = (args.get("query", "") or 
                                       args.get("sql_query", "") or 
                                       args.get("natural_query", "") or
                                       args.get("description", ""))
                            
                            # Only fallback to str() if still empty
                            if not query:
                                query = str(args)[:100]
                            break
                    if query:
                        break
            
            # Store in memory (using sync wrapper for non-async routing context)
            # ✅ FIX: Pass full args dict, not just {"query": query}
            # Note: add_tool_call_sync now handles persistence internally when auto_persist is enabled
            add_tool_start = time.perf_counter()
            state.memory.add_tool_call_sync(tool_name, args, result)
            add_tool_duration = (time.perf_counter() - add_tool_start) * 1000
            logger.info(f"⏱️  [ROUTER]   add_tool_call_sync: {add_tool_duration:.1f}ms")

            # ✅ OPTIMIZATION: Mark that we just persisted to avoid redundant writes in FINISH node
            import time as time_module
            state.runtime_metadata["last_memory_persist"] = time_module.time()
            logger.info(f"⏱️  [ROUTER]   Set last_memory_persist timestamp to avoid redundant DB writes")

            memory_duration = (time.perf_counter() - memory_start) * 1000
            logger.info(f"⏱️  [ROUTER]   Total memory storage: {memory_duration:.1f}ms")

        except Exception as e:
            logger.error(f"❌ [ROUTER] Failed to store tool result: {e}")

    # Check for file approval needs
    approval_check_start = time.perf_counter()
    try:
        result = json.loads(last_tool_message.content)

        if result.get("needs_approval"):
            logger.info(f"🔀 [Router] TOOLS → CHECK_FILE_APPROVAL ({last_tool_message.name})")
            return "check_file_approval"
    except (json.JSONDecodeError, AttributeError):
        pass

    approval_check_duration = (time.perf_counter() - approval_check_start) * 1000
    logger.info(f"⏱️  [ROUTER]   Approval check: {approval_check_duration:.1f}ms")

    routing_duration = (time.perf_counter() - routing_start) * 1000
    logger.info(f"⏱️  [ROUTER] route_after_tools() completed in {routing_duration:.1f}ms")
    logger.info("🔀 [Router] TOOLS → ASSESS")
    return "assess"


def route_classification_aware(state: State) -> Literal["direct_act", "simple_plan", "plan", "act"]:
    """Unified routing function that considers both classification and plan state."""
    if state.plan is not None:
        logger.info("🔀 [Router] UNIFIED → ACT (plan exists)")
        return "act"
    
    complexity_level = state.runtime_metadata.get("complexity_level")
    
    if complexity_level == "direct":
        logger.info("🔀 [Router] UNIFIED → DIRECT_ACT (complexity=direct)")
        return "direct_act"
    elif complexity_level == "simple_plan":
        logger.info("🔀 [Router] UNIFIED → SIMPLE_PLAN (complexity=simple_plan)")
        return "simple_plan"
    elif complexity_level == "complex_plan":
        logger.info("🔀 [Router] UNIFIED → PLAN (complexity=complex_plan)")
        return "plan"
    else:
        logger.info(f"🔀 [Router] UNIFIED → PLAN (complexity={complexity_level})")
        return "plan"