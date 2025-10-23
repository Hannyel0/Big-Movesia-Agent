"""Enhanced assessment node with detailed error context extraction and forwarding."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.state import (
    AssessmentOutcome,
    State,
    StructuredAssessment,
    StepStatus,
    ExecutionPlan,
    PlanStep,
)
from react_agent.narration import NarrationEngine, StreamingNarrator
from react_agent.utils import get_message_text, get_model
from react_agent.memory import get_memory_insights

# Set up logging
logger = logging.getLogger(__name__)

# Initialize narration components
narration_engine = NarrationEngine()


def _extract_detailed_tool_errors(tool_result: dict, tool_name: str) -> Dict[str, Any]:
    """
    Extract detailed error information from tool results for context-aware recovery.
    
    This is crucial for the error recovery system to understand what specifically
    went wrong rather than just getting a generic error category.
    """
    detailed_context = {
        "tool_name": tool_name,
        "raw_result": tool_result,
        "extracted_errors": [],
        "error_count": 0,
        "error_summary": "",
        "specific_issues": []
    }
    
    if tool_name == "compile_and_test":
        # Extract compilation errors - this is the key fix for your issue
        compilation_errors = tool_result.get("compilation_errors", [])
        warnings_data = tool_result.get("warnings", [])
        
        # Process compilation errors
        if isinstance(compilation_errors, list):
            for error in compilation_errors:
                if isinstance(error, dict):
                    detailed_context["extracted_errors"].append({
                        "type": "compilation_error",
                        "file": error.get("file", ""),
                        "line": error.get("line", 0),
                        "error": error.get("error", ""),
                        "severity": "error"
                    })
        
        # Also check nested error structures (your actual tool output format)
        if "details" in tool_result:
            details = tool_result["details"]
            if isinstance(details, dict):
                # Check for errors in details
                detail_errors = details.get("errors", [])
                if isinstance(detail_errors, list):
                    for error in detail_errors:
                        if isinstance(error, dict):
                            detailed_context["extracted_errors"].append({
                                "type": "compilation_error",
                                "file": error.get("file", ""),
                                "line": error.get("line", 0),
                                "error": error.get("error", ""),
                                "severity": "error"
                            })
                
                # Check for warnings in details (when warnings is an int count at top level)
                detail_warnings = details.get("warnings", [])
                if isinstance(detail_warnings, list):
                    for warning in detail_warnings:
                        if isinstance(warning, str):
                            # Handle string format: "Unused variable 'tempVar' in PlayerController.cs line 23"
                            detailed_context["extracted_errors"].append({
                                "type": "compilation_warning",
                                "file": "extracted from message",
                                "line": 0,
                                "error": warning,
                                "severity": "warning"
                            })
                        elif isinstance(warning, dict):
                            # Handle dict format
                            detailed_context["extracted_errors"].append({
                                "type": "compilation_warning",
                                "file": warning.get("file", ""),
                                "line": warning.get("line", 0),
                                "error": warning.get("warning", ""),
                                "severity": "warning"
                            })
        
        # Process warnings (handle both int count and array formats)
        if isinstance(warnings_data, list):
            # warnings_data is an array of warning objects
            for warning in warnings_data:
                if isinstance(warning, dict):
                    detailed_context["extracted_errors"].append({
                        "type": "compilation_warning",
                        "file": warning.get("file", ""),
                        "line": warning.get("line", 0),
                        "error": warning.get("warning", ""),
                        "severity": "warning"
                    })
                elif isinstance(warning, str):
                    # Handle string warnings
                    detailed_context["extracted_errors"].append({
                        "type": "compilation_warning",
                        "file": "extracted from message",
                        "line": 0,
                        "error": warning,
                        "severity": "warning"
                    })
        elif isinstance(warnings_data, int):
            # warnings_data is just a count - actual warnings might be in details
            logger.info(f"Found {warnings_data} warnings (count only, details might be nested)")
        
        # FALLBACK: Check for compilation errors at top level if none found yet
        if not detailed_context["extracted_errors"]:
            # Look for errors/warnings in different possible locations
            top_level_errors = tool_result.get("errors", 0)
            if isinstance(top_level_errors, int) and top_level_errors > 0:
                # Create a generic error entry
                detailed_context["extracted_errors"].append({
                    "type": "compilation_error",
                    "file": "multiple files",
                    "line": 0,
                    "error": f"Compilation failed with {top_level_errors} errors",
                    "severity": "error"
                })
        
        # Count actual errors (not warnings)
        error_count = len([e for e in detailed_context["extracted_errors"] if e["severity"] == "error"])
        detailed_context["error_count"] = error_count
        
        # Create summary for logging/diagnosis
        if detailed_context["extracted_errors"]:
            error_files = set()
            error_types = set()
            
            for error in detailed_context["extracted_errors"]:
                if error["severity"] == "error":
                    file_name = error["file"].split("/")[-1] if error["file"] else "unknown"
                    error_files.add(file_name)
                    
                    error_msg = error["error"].lower()
                    if "inputsystem" in error_msg:
                        error_types.add("InputSystem issues")
                    elif "charactercontroller" in error_msg:
                        error_types.add("CharacterController issues")
                    elif "namespace" in error_msg or "using" in error_msg:
                        error_types.add("namespace issues")
                    elif "not found" in error_msg or "does not exist" in error_msg:
                        error_types.add("missing references")
                    elif "cs0103" in error_msg or "cs0246" in error_msg:
                        error_types.add("compilation errors")
            
            if error_types:
                detailed_context["error_summary"] = f"{error_count} errors in {len(error_files)} files: {', '.join(error_types)}"
            else:
                detailed_context["error_summary"] = f"{error_count} compilation errors found"
        else:
            detailed_context["error_summary"] = "No specific errors extracted"
        
        logger.info(f"Extracted {error_count} compilation errors from tool result")
        
    elif tool_name in ["read_file", "write_file", "modify_file", "delete_file", "move_file"]:
        # Extract file operation errors
        error_msg = tool_result.get("error", "")
        if error_msg:
            file_path = tool_result.get("file_path", "")
            detailed_context["extracted_errors"].append({
                "type": "file_operation_error",
                "operation": tool_name,
                "file": file_path,
                "error": error_msg,
                "severity": "error"
            })
            detailed_context["error_count"] = 1
            detailed_context["error_summary"] = f"{tool_name} error: {error_msg[:50]}"
    
    elif tool_name == "web_search":
        # Extract web search errors
        error_msg = tool_result.get("error", "")
        if error_msg:
            detailed_context["extracted_errors"].append({
                "type": "web_search_error",
                "query": tool_result.get("query", ""),
                "error": error_msg,
                "severity": "error"
            })
            detailed_context["error_count"] = 1
            detailed_context["error_summary"] = f"Web search error: {error_msg[:50]}"
    
    elif tool_name == "search_project":
        # Extract project search errors
        error_msg = tool_result.get("error", "")
        if error_msg:
            detailed_context["extracted_errors"].append({
                "type": "search_project_error",
                "query": tool_result.get("query_description", ""),
                "error": error_msg,
                "severity": "error"
            })
            detailed_context["error_count"] = 1
            detailed_context["error_summary"] = f"Project search error: {error_msg[:50]}"
    
    elif tool_name == "code_snippets":
        # Extract code search errors
        error_msg = tool_result.get("error", "")
        if error_msg:
            detailed_context["extracted_errors"].append({
                "type": "code_search_error",
                "query": tool_result.get("query", ""),
                "error": error_msg,
                "severity": "error"
            })
            detailed_context["error_count"] = 1
            detailed_context["error_summary"] = f"Code search error: {error_msg[:50]}"
    
    else:
        # Generic error extraction for other tools
        error_msg = tool_result.get("error", "")
        if error_msg:
            detailed_context["extracted_errors"].append({
                "type": "tool_error",
                "tool": tool_name,
                "error": error_msg,
                "severity": "error"
            })
            detailed_context["error_count"] = 1
            detailed_context["error_summary"] = f"{tool_name} error: {error_msg[:50]}"
    
    return detailed_context


def _should_trigger_micro_retry(tool_result: dict, tool_name: str, retry_count: int) -> bool:
    """
    Check if this error should trigger micro-retry instead of normal retry.
    
    This function implements Tier 0 detection - identifying transient errors
    that can be resolved with immediate retry rather than planning.
    """
    # Don't micro-retry if we've already tried multiple times
    if retry_count >= 2:
        return False
        
    if not tool_result.get("success", True):
        error_message = tool_result.get("error", "").lower()
        error_category = tool_result.get("error_category", "").lower()
        
        # Check for transient error categories
        transient_categories = [
            "network_error", "tool_malfunction", "timeout", 
            "rate_limit", "service_unavailable"
        ]
        
        if error_category in transient_categories:
            logger.info(f"Detected transient error category: {error_category}")
            return True
        
        # Check for transient patterns in error message
        transient_patterns = [
            "network", "timeout", "connection", "rate limit", 
            "temporary", "try again", "service unavailable",
            "internal server error", "502", "503", "504",
            "connection reset", "connection refused", "dns"
        ]
        
        if any(pattern in error_message for pattern in transient_patterns):
            logger.info(f"Detected transient error pattern in: {error_message[:100]}")
            return True
    
    return False


def _is_tool_result_successful(tool_result: dict, tool_name: str) -> bool:
    """Check if a tool result indicates success based on tool-specific criteria."""
    if not tool_result:
        return False
    
    # Check explicit success flag first
    if "success" in tool_result:
        return tool_result["success"] is True
    
    # Tool-specific success checks for production tools
    if tool_name == "search_project":
        return bool(tool_result.get("results")) and tool_result.get("result_count", 0) > 0
    elif tool_name == "code_snippets":
        return bool(tool_result.get("snippets")) and tool_result.get("total_found", 0) > 0
    elif tool_name in ["read_file", "write_file", "modify_file", "delete_file", "move_file"]:
        return tool_result.get("success", False)
    elif tool_name == "web_search":
        return bool(tool_result.get("results")) and len(tool_result.get("results", [])) > 0
    
    return len(tool_result) > 1


async def assess(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Enhanced assessment with detailed error context extraction and micro-retry detection."""
    logger.info(f"🔍 [ASSESS] ===== STARTING ASSESSMENT =====")
    
    # ✅ FIX: Handle direct actions (no plan)
    if not state.plan:
        logger.info(f"🔍 [ASSESS] Direct action mode - no plan to assess")
        logger.info(f"🔍 [ASSESS] Creating implicit success assessment")
        
        # Extract the tool result
        tool_result = None
        tool_name = "unknown"
        
        for msg in reversed(state.messages[-10:]):
            if isinstance(msg, ToolMessage):
                try:
                    content = get_message_text(msg)
                    tool_result = json.loads(content) if content else {}
                    tool_name = msg.name or "unknown"
                    logger.info(f"🔍 [ASSESS] Found tool result from: {tool_name}")
                    break
                except json.JSONDecodeError:
                    tool_result = {"message": content}
                    break
        
        # ❌ REMOVED: Duplicate tool call recording (already done in routing.py)
        # Tool results are captured by route_after_tools() before reaching assess
        
        # ✅ Verify memory state for logging
        if state.memory and hasattr(state.memory, 'working_memory'):
            recent_tools = state.memory.working_memory.recent_tool_results
            logger.info(f"🧠 [ASSESS] Working memory now has {len(recent_tools)} tool results")
        
        # Create success assessment for direct actions
        assessment = AssessmentOutcome(
            outcome="success",
            reason=f"Direct action completed successfully with {tool_name}",
            confidence=0.9
        )
        
        logger.info(f"🔍 [ASSESS] ===== ASSESSMENT COMPLETE (DIRECT) =====")
        
        # ✅ FIX: Track completed step for direct actions too
        updated_completed_steps = list(state.completed_steps) if state.completed_steps else []
        if state.step_index not in updated_completed_steps:
            updated_completed_steps.append(state.step_index)
        
        return {
            "current_assessment": assessment,
            "completed_steps": updated_completed_steps,
            "total_assessments": state.total_assessments + 1
        }
    
    # Original assessment logic for planned executions
    logger.info(f"🔍 [ASSESS] Plan mode - assessing step {state.step_index + 1}/{len(state.plan.steps)}")
    
    context = runtime.context
    model = get_model(context.assessment_model or context.model)
    
    if state.step_index >= len(state.plan.steps):
        logger.warning("🔍 [ASSESS] Invalid step index - returning empty")
        return {}
    
    current_step = state.plan.steps[state.step_index]
    logger.info(f"🔍 [ASSESS] Current step: {current_step.description}")
    logger.info(f"🔍 [ASSESS] Tool: {current_step.tool_name}")
    
    # Extract the tool result from the last ToolMessage
    tool_result = None
    tool_name = current_step.tool_name
    
    for msg in reversed(state.messages[-10:]):
        if isinstance(msg, ToolMessage):
            try:
                content = get_message_text(msg)
                tool_result = json.loads(content) if content else {}
                break
            except json.JSONDecodeError:
                tool_result = {"message": content}
                break
    
    # ✅ ADD: Log tool result analysis
    if tool_result:
        logger.info(f"🔍 [ASSESS] Tool result analysis:")
        logger.info(f"🔍 [ASSESS]   Success: {tool_result.get('success', 'N/A')}")
        logger.info(f"🔍 [ASSESS]   Tool: {tool_name}")
        if not tool_result.get("success", True):
            error_msg = tool_result.get('error', 'No error message')
            logger.warning(f"🔍 [ASSESS]   Error: {error_msg[:100]}")
    else:
        logger.warning("🔍 [ASSESS] No tool result found - creating default assessment")
        assessment = AssessmentOutcome(
            outcome="retry",
            reason="No tool result found",
            confidence=0.5
        )
        return {
            "current_assessment": assessment,
            "total_assessments": state.total_assessments + 1
        }
    
    # ❌ REMOVED: Duplicate tool call recording (already done in routing.py)
    # Tool results are captured by route_after_tools() before reaching assess
    
    # ✅ Verify memory state for logging
    if state.memory and hasattr(state.memory, 'working_memory'):
        recent_tools = state.memory.working_memory.recent_tool_results
        logger.info(f"🧠 [ASSESS] Working memory has {len(recent_tools)} tool results")
    
    # ENHANCED: Extract detailed error context before any decisions
    detailed_error_context = _extract_detailed_tool_errors(tool_result, tool_name)
    
    # TIER 0: Check for micro-retry opportunity FIRST
    if _should_trigger_micro_retry(tool_result, tool_name, state.retry_count):
        error_message = tool_result.get("error", "Unknown transient error")
        logger.info(f"🔍 [ASSESS] Triggering micro-retry for transient error: {error_message[:80]}")
        
        assessment = AssessmentOutcome(
            outcome="retry",  # Keep as retry, but flag for micro-retry
            reason=f"Transient error in {tool_name}: {error_message}",
            confidence=0.9
        )
        
        return {
            "current_assessment": assessment,
            "should_micro_retry": True,  # NEW FLAG for micro-retry
            "total_assessments": state.total_assessments + 1,
            "runtime_metadata": {
                **state.runtime_metadata,
                "micro_retry_triggered": True,
                "micro_retry_reason": error_message[:100]
            }
        }
    
    # Normal assessment flow continues...
    # Check if tool result indicates success
    tool_looks_successful = _is_tool_result_successful(tool_result, tool_name)
    logger.info(f"🔍 [ASSESS] Tool success check: {tool_looks_successful}")
    
    # Create assessment based on tool result
    if tool_looks_successful:
        assessment = AssessmentOutcome(
            outcome="success",
            reason=f"Tool {tool_name} completed successfully with expected output",
            confidence=0.9
        )
        logger.info("🔍 [ASSESS] Created SUCCESS assessment")
    else:
        # Check for explicit error in tool result
        tool_error = tool_result.get("error", "")
        if tool_error or detailed_error_context["error_count"] > 0:
            # Use detailed error context for better assessment
            if detailed_error_context["error_count"] > 0:
                error_summary = detailed_error_context["error_summary"]
                reason = f"Tool {tool_name} failed with {detailed_error_context['error_count']} errors: {error_summary}"
                logger.warning(f"🔍 [ASSESS] Tool failed with {detailed_error_context['error_count']} errors")
            else:
                reason = f"Tool {tool_name} failed: {tool_error}"
                logger.warning(f"🔍 [ASSESS] Tool failed: {tool_error[:80]}")
            
            assessment = AssessmentOutcome(
                outcome="retry",
                reason=reason,
                confidence=0.8
            )
            
            # Check if this should trigger error recovery (non-transient errors)
            if _should_trigger_error_recovery(tool_result, tool_error, detailed_error_context):
                logger.info(f"🔍 [ASSESS] Triggering error recovery for {detailed_error_context['error_count']} non-transient errors")
                
                # ENHANCED: Include detailed error context in error_context
                error_context = {
                    "tool_result": tool_result,
                    "tool_name": tool_name,
                    "step_description": current_step.description,
                    "assessment": assessment,
                    "retry_count": state.retry_count,
                    "detailed_errors": detailed_error_context  # NEW: Include all detailed error info
                }
                
                return {
                    "current_assessment": assessment,
                    "needs_error_recovery": True,
                    "error_context": error_context,
                    "total_assessments": state.total_assessments + 1
                }
        else:
            assessment = AssessmentOutcome(
                outcome="success",  # If no explicit error, assume success
                reason=f"Tool {tool_name} completed without explicit errors",
                confidence=0.7
            )
            logger.info("🔍 [ASSESS] Created SUCCESS assessment (no explicit error)")
    
    # ✅ FIX: Mark step as SUCCEEDED in plan when assessment is successful
    updated_plan = None
    if assessment.outcome == "success" and state.plan and state.step_index < len(state.plan.steps):
        updated_steps = list(state.plan.steps)
        current = updated_steps[state.step_index]
        updated_steps[state.step_index] = PlanStep(
            description=current.description,
            tool_name=current.tool_name,
            success_criteria=current.success_criteria,
            dependencies=current.dependencies,
            status=StepStatus.SUCCEEDED,  # ✅ Mark as SUCCEEDED
            attempts=current.attempts,
            error_messages=current.error_messages
        )
        
        updated_plan = ExecutionPlan(
            goal=state.plan.goal,
            steps=updated_steps,
            metadata=state.plan.metadata
        )
        logger.info(f"🔍 [ASSESS] Marked step {state.step_index} as SUCCEEDED")
    
    # Prepare messages with narration
    messages_to_add = []
    
    # Handle completion or transitions
    if assessment.outcome == "success":
        next_index = state.step_index + 1
        is_final_step = next_index >= len(state.plan.steps)
        
        if is_final_step:
            completion_summary = await _generate_completion_summary(state, model, context)
            messages_to_add.append(AIMessage(content=completion_summary))
            logger.info("🔍 [ASSESS] Added completion summary - final step")
        elif next_index < len(state.plan.steps):
            next_step = state.plan.steps[next_index]
            transition = f"✅ Step completed successfully! Moving on to: {next_step.description}"
            messages_to_add.append(AIMessage(content=transition))
            logger.info("🔍 [ASSESS] Added transition message to next step")
    
    # ✅ MEMORY: Add assessment to memory
    if state.memory:
        state.memory.add_assessment({
            "step_index": state.step_index,
            "outcome": assessment.outcome,
            "reason": assessment.reason
        })
        
        # ✅ MEMORY: Track successful tool usage in patterns
        if assessment.outcome == "success" and state.memory:
            logger.info(f"🧠 [ASSESS] Success recorded: {current_step.tool_name} works for '{current_step.description[:40]}'")
        
        # Add errors to memory
        if assessment.outcome == "retry" and detailed_error_context["error_count"] > 0:
            state.memory.add_error({
                "step_index": state.step_index,
                "errors": detailed_error_context["extracted_errors"]
            })
            logger.info(f"🧠 [ASSESS] Recorded {detailed_error_context['error_count']} errors in memory")
    
    # ✅ MEMORY: Check insights before retrying
    if assessment.outcome == "retry" and state.memory:
        insights = get_memory_insights(state)
        
        # Check if we've seen this pattern fail before
        if insights["relevant_patterns"]:
            for pattern in insights["relevant_patterns"]:
                if pattern["success_rate"] < 0.3:  # This approach usually fails
                    logger.info(f"🧠 [ASSESS] Memory shows low success rate ({pattern['success_rate']:.0%}) for pattern: {pattern['context'][:60]}")
                    
                    # Trigger replanning instead of retry
                    return {
                        "current_assessment": assessment,
                        "needs_error_recovery": True,  # Force better approach
                        "error_context": {
                            "tool_result": tool_result,
                            "tool_name": tool_name,
                            "step_description": current_step.description,
                            "assessment": assessment,
                            "retry_count": state.retry_count,
                            "detailed_errors": detailed_error_context,
                            "memory_insight": f"Pattern '{pattern['context']}' has {pattern['success_rate']:.0%} success rate"
                        },
                        "total_assessments": state.total_assessments + 1
                    }
    
    # ✅ ADD: Log assessment decision
    logger.info(f"🔍 [ASSESS] Assessment outcome: {assessment.outcome}")
    logger.info(f"🔍 [ASSESS] Reason: {assessment.reason[:100]}")
    logger.info(f"🔍 [ASSESS] Confidence: {assessment.confidence}")
    
    if assessment.outcome == "retry":
        logger.info(f"🔍 [ASSESS] Retry count: {state.retry_count + 1}/{state.max_retries_per_step}")
    
    result = {
        "current_assessment": assessment,
        "total_assessments": state.total_assessments + 1
    }
    
    # ✅ FIX: Update both plan status and completed_steps tracking
    if assessment.outcome == "success":
        # Track in completed_steps list
        updated_completed_steps = list(state.completed_steps) if state.completed_steps else []
        if state.step_index not in updated_completed_steps:
            updated_completed_steps.append(state.step_index)
            result["completed_steps"] = updated_completed_steps
            logger.info(f"🔍 [ASSESS] Added step {state.step_index} to completed_steps list")
        
        # Update plan with SUCCEEDED status
        if updated_plan:
            result["plan"] = updated_plan
    
    if messages_to_add:
        result["messages"] = messages_to_add
    
    logger.info(f"🔍 [ASSESS] ===== ASSESSMENT COMPLETE =====")
    
    return result


def _should_trigger_error_recovery(tool_result: dict, error_message: str, detailed_error_context: Dict[str, Any]) -> bool:
    """Determine if this error should trigger error recovery (enhanced with detailed context)."""
    if not tool_result.get("success", True):  # Explicit failure
        error_lower = error_message.lower()
        
        # Critical error patterns that need recovery (non-transient)
        critical_patterns = [
            "not found", "missing", "does not exist",
            "compilation failed", "build failed", 
            "dependency", "package", "import error",
            "configuration error", "invalid parameter",
            "permission denied", "access denied",
            "file not found", "directory not found",
            "syntax error", "reference error",
            "cs0103", "cs0246"  # Specific C# compilation errors
        ]
        
        # Only trigger recovery for non-transient errors
        transient_patterns = [
            "network", "timeout", "connection", "rate limit",
            "temporary", "try again", "service unavailable"
        ]
        
        # Check if it's a critical error but not transient
        is_critical = any(pattern in error_lower for pattern in critical_patterns)
        is_transient = any(pattern in error_lower for pattern in transient_patterns)
        
        # Also consider error count from detailed context
        has_multiple_errors = detailed_error_context.get("error_count", 0) > 0
        
        return (is_critical or has_multiple_errors) and not is_transient
    
    return False


async def _generate_completion_summary(state: State, model, context: Context) -> str:
    """Generate a completion summary using the LLM based on what was actually accomplished."""
    
    completed_steps_summary = []
    for i, step in enumerate(state.plan.steps):
        if i <= state.step_index:
            completed_steps_summary.append(f"Step {i+1}: {step.description} (using {step.tool_name})")
    
    recent_tool_results = []
    for msg in reversed(state.messages[-10:]):
        if isinstance(msg, ToolMessage):
            try:
                result = json.loads(get_message_text(msg))
                tool_name = msg.name or 'unknown'
                if _is_tool_result_successful(result, tool_name):
                    if tool_name == "file_operation":
                        operation = result.get('operation', 'operation')
                        file_path = result.get('file_path', 'file')
                        recent_tool_results.append(f"File {operation}: {file_path}")
                    elif tool_name == "search_project":
                        count = result.get('result_count', 0)
                        recent_tool_results.append(f"Found {count} project items")
                    elif tool_name == "code_snippets":
                        count = result.get('total_found', 0)
                        recent_tool_results.append(f"Found {count} code snippets")
                    elif tool_name == "web_search":
                        count = len(result.get('results', []))
                        recent_tool_results.append(f"Found {count} web resources")
                    elif tool_name == "unity_docs":
                        count = len(result.get('results', []))
                        recent_tool_results.append(f"Found {count} Unity documentation pages")
                    else:
                        recent_tool_results.append(f"{tool_name}: {result.get('message', 'completed successfully')}")
            except:
                recent_tool_results.append(f"{msg.name}: completed")
    
    completion_prompt = f"""You have just completed all steps for the goal: "{state.plan.goal}"
Completed Steps:
{chr(10).join(completed_steps_summary)}

Key Results:
{chr(10).join(recent_tool_results[-5:]) if recent_tool_results else "All steps completed successfully"}

Generate a brief, professional completion message (1-2 sentences) that starts with "Perfect!" and summarizes what was accomplished."""

    try:
        completion_messages = [
            {"role": "system", "content": "You are providing a brief, professional completion summary for a Unity development task. Be concise, specific, and confident."},
            {"role": "user", "content": completion_prompt}
        ]
        
        completion_response = await model.ainvoke(completion_messages)
        completion_summary = get_message_text(completion_response).strip()
        
        if len(completion_summary) < 20 or not completion_summary.lower().startswith(('perfect', 'excellent', 'great', 'done', 'completed', 'success')):
            return f"Perfect! I've successfully completed your request: {state.plan.goal}"
        
        return completion_summary
        
    except Exception as e:
        return f"Perfect! I've successfully completed all steps for: {state.plan.goal}"