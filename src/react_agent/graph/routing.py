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
    logger.info(f"🔀 [ROUTER] ===== ROUTING AFTER ASSESS =====")
    
    # Log current state
    if state.current_assessment:
        logger.info(f"🔀 [ROUTER] Assessment: {state.current_assessment.outcome}")
        logger.info(f"🔀 [ROUTER] Confidence: {state.current_assessment.confidence}")
    else:
        logger.info(f"🔀 [ROUTER] Assessment: None")
    
    if state.plan and state.plan.steps:
        logger.info(f"🔀 [ROUTER] Step: {state.step_index + 1}/{len(state.plan.steps)}")
        logger.info(f"🔀 [ROUTER] Retry count: {state.retry_count}/{state.max_retries_per_step}")
    
    # Safety checks
    if not state.current_assessment:
        logger.info("🔀 [ROUTER] Decision: FINISH")
        logger.info("🔀 [ROUTER] Reason: No assessment available")
        return "finish"
    
    if not state.plan or not state.plan.steps:
        logger.info("🔀 [ROUTER] Decision: FINISH")
        logger.info("🔀 [ROUTER] Reason: No plan or steps available")
        return "finish"
    
    # NEW: Check for micro-retry flag first (highest priority for transient errors)
    if getattr(state, "should_micro_retry", False):
        logger.info("🔀 [ROUTER] Decision: MICRO_RETRY")
        logger.info("🔀 [ROUTER] Reason: Transient error detected, attempting immediate retry")
        return "micro_retry"
    
    # ABSOLUTE PRIORITY: Check if we're on or past the last step
    last_step_index = len(state.plan.steps) - 1
    is_on_or_past_last_step = state.step_index >= last_step_index
    
    # If we're on the last step AND it succeeded, ALWAYS finish
    if is_on_or_past_last_step and state.current_assessment.outcome == "success":
        logger.info("🔀 [ROUTER] Decision: FINISH")
        logger.info(f"🔀 [ROUTER] Reason: Last step (step {state.step_index + 1}/{len(state.plan.steps)}) completed successfully")
        return "finish"
    
    # If we somehow went past the last step, force finish
    if state.step_index >= len(state.plan.steps):
        logger.info("🔀 [ROUTER] Decision: FINISH")
        logger.info(f"🔀 [ROUTER] Reason: Step index ({state.step_index + 1}) exceeds plan length ({len(state.plan.steps)})")
        return "finish"
    
    # Check for error recovery need (only if not completing)
    if getattr(state, "needs_error_recovery", False) or state.runtime_metadata.get("needs_error_recovery"):
        logger.info("🔀 [ROUTER] Decision: ERROR_RECOVERY")
        logger.info("🔀 [ROUTER] Reason: Error recovery explicitly requested by assessment")
        return "error_recovery"
    
    # Handle success on non-final steps
    if state.current_assessment.outcome == "success":
        logger.info("🔀 [ROUTER] Decision: ADVANCE_STEP")
        logger.info(f"🔀 [ROUTER] Reason: Step {state.step_index + 1}/{len(state.plan.steps)} succeeded, advancing to next step")
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
                    logger.info("🔀 [ROUTER] Decision: ERROR_RECOVERY")
                    logger.info(f"🔀 [ROUTER] Reason: Memory shows low success rate ({success_rate:.0%}) for this pattern")
                    logger.info(f"🔀 [ROUTER] Pattern: {pattern.get('context', 'unknown')[:60]}")
                    print(f"🧠 [Router] Memory shows low success rate ({success_rate:.0%}) for this pattern")
                    print(f"   Pattern: {pattern.get('context', 'unknown')}")
                    print(f"   Skipping retry, routing to error_recovery")
                    return "error_recovery"
        
        if state.retry_count >= state.max_retries_per_step:
            logger.info("🔀 [ROUTER] Decision: ERROR_RECOVERY")
            logger.info(f"🔀 [ROUTER] Reason: Max retries reached ({state.retry_count}/{state.max_retries_per_step})")
            return "error_recovery"
        
        logger.info("🔀 [ROUTER] Decision: INCREMENT_RETRY")
        logger.info(f"🔀 [ROUTER] Reason: Assessment suggests retry (attempt {state.retry_count + 1}/{state.max_retries_per_step})")
        return "increment_retry"
    
    # Handle blocked steps
    if state.current_assessment.outcome == "blocked":
        logger.info("🔀 [ROUTER] Decision: ERROR_RECOVERY")
        logger.info("🔀 [ROUTER] Reason: Step is blocked and cannot proceed")
        return "error_recovery"
    
    # Fallback to finish
    logger.info("🔀 [ROUTER] Decision: FINISH")
    logger.info(f"🔀 [ROUTER] Reason: Fallback - unhandled outcome ({state.current_assessment.outcome})")
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
    logger.info("🔀 [ROUTER] ===== ROUTING AFTER TOOLS =====")
    
    # ✅ CRITICAL: Capture tool result in memory FIRST
    last_tool_message = None
    for msg in reversed(state.messages):
        if isinstance(msg, ToolMessage):
            last_tool_message = msg
            break
    
    if not last_tool_message:
        logger.warning("🔀 [ROUTER] No tool message found")
        return "assess"
    
    logger.info(f"🔀 [ROUTER] Found tool message from: {last_tool_message.name}")
    
    # ✅ CRITICAL: Store tool result in memory
    if state.memory:
        try:
            result = json.loads(last_tool_message.content)
            tool_name = last_tool_message.name or "unknown"
            
            logger.info(f"🧠 [ROUTER] Storing tool result in memory")
            logger.info(f"🧠 [ROUTER]   Tool: {tool_name}")
            logger.info(f"🧠 [ROUTER]   Success: {result.get('success', 'N/A')}")
            
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
            
            logger.info(f"🧠 [ROUTER]   Query: {query[:80]}")
            logger.info(f"🧠 [ROUTER]   Args keys: {list(args.keys())}")
            
            # Store in memory (using sync wrapper for non-async routing context)
            # ✅ FIX: Pass full args dict, not just {"query": query}
            # Note: add_tool_call_sync now handles entity extraction and focus updates automatically
            state.memory.add_tool_call_sync(tool_name, args, result)
            
            # ✅ FIX: Persist immediately to survive interrupts
            if state.memory and state.memory.auto_persist:
                try:
                    # Get entity count from semantic memory
                    entity_count = len(state.memory.semantic_memory.entity_knowledge)
                    if entity_count > 0:
                        state.memory._persist_to_database_sync()
                        logger.info(f"🧠 [ROUTER]   ✅ Persisted {entity_count} entities to database")
                except Exception as e:
                    logger.warning(f"🧠 [ROUTER]   ⚠️ Failed to persist entities: {e}")
            
            # ✅ VERIFY STORAGE
            if hasattr(state.memory, 'working_memory'):
                recent_tools = state.memory.working_memory.recent_tool_results
                logger.info(f"✅ [ROUTER] Memory verification:")
                logger.info(f"   Total stored: {len(recent_tools)} tool results")
                
                if recent_tools:
                    # Show last 3 tool results
                    for i, tool in enumerate(recent_tools[-3:], 1):
                        logger.info(f"   {i}. {tool['tool_name']}: {tool['summary'][:60]}")
                    
                    # Print to console for visibility
                    print(f"\n✅ [ROUTER] Tool result stored in memory:")
                    print(f"   Tool: {tool_name}")
                    print(f"   Total in memory: {len(recent_tools)}")
                    print(f"   Latest: {recent_tools[-1]['summary']}")
                else:
                    logger.warning(f"⚠️ [ROUTER] Tool result was stored but recent_tool_results is empty!")
                    print(f"⚠️ [ROUTER] WARNING: Tool result stored but recent_tool_results is empty!")
            else:
                logger.warning(f"⚠️ [ROUTER] state.memory has no working_memory attribute!")
                print(f"⚠️ [ROUTER] WARNING: state.memory has no working_memory attribute!")
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ [ROUTER] Failed to parse tool result as JSON: {e}")
            print(f"❌ [ROUTER] ERROR: Failed to parse tool result as JSON: {e}")
        except Exception as e:
            logger.error(f"❌ [ROUTER] Failed to store tool result: {e}", exc_info=True)
            print(f"❌ [ROUTER] ERROR: Failed to store tool result: {e}")
    else:
        logger.warning(f"⚠️ [ROUTER] No memory available to store tool result")
        print(f"⚠️ [ROUTER] WARNING: No memory available to store tool result")
    
    # Check for file approval needs
    try:
        result = json.loads(last_tool_message.content)
        
        if result.get("needs_approval"):
            logger.info(f"🔀 [ROUTER] Decision: CHECK_FILE_APPROVAL")
            logger.info(f"🔀 [ROUTER]   Tool: {last_tool_message.name}")
            logger.info(f"🔀 [ROUTER]   Operation: {result.get('approval_data', {}).get('operation')}")
            return "check_file_approval"
    except (json.JSONDecodeError, AttributeError) as e:
        logger.warning(f"🔀 [ROUTER] Could not check approval needs: {e}")
    
    logger.info("🔀 [ROUTER] Decision: ASSESS")
    logger.info(f"🔀 [ROUTER] ===== ROUTING COMPLETE =====")
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