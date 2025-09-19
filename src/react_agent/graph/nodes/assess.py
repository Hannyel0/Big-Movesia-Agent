"""Fixed assessment node with proper error recovery integration."""

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
)
from react_agent.narration import NarrationEngine, StreamingNarrator
from react_agent.utils import get_message_text, get_model

# Set up logging
logger = logging.getLogger(__name__)

# Initialize narration components
narration_engine = NarrationEngine()


def _is_tool_result_successful(tool_result: dict, tool_name: str) -> bool:
    """Check if a tool result indicates success based on tool-specific criteria."""
    if not tool_result:
        return False
    
    # Check explicit success flag first
    if "success" in tool_result:
        return tool_result["success"] is True
    
    # Tool-specific success checks for simulated tools
    if tool_name == "create_asset":
        return bool(tool_result.get("asset_type") and tool_result.get("name"))
    elif tool_name == "get_script_snippets":
        return bool(tool_result.get("snippets") or tool_result.get("available_snippets") or tool_result.get("snippets_by_script"))
    elif tool_name == "write_file":
        return bool(tool_result.get("file_path") and tool_result.get("size_bytes") is not None)
    elif tool_name == "compile_and_test":
        return "compilation_time" in tool_result or "errors" in tool_result
    elif tool_name == "search":
        return bool(tool_result.get("result")) and len(tool_result.get("result", [])) > 0
    elif tool_name == "get_project_info":
        return bool(tool_result.get("engine") or tool_result.get("project_name"))
    elif tool_name == "scene_management":
        return bool(tool_result.get("action") and tool_result.get("scene_name"))
    elif tool_name == "edit_project_config":
        return bool(tool_result.get("config_section"))
    
    return len(tool_result) > 1


async def assess(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Fixed assessment with proper error handling and recovery integration."""
    logger.info(f"=== ASSESS DEBUG START ===")
    logger.info(f"Step index: {state.step_index}")
    logger.info(f"Plan steps: {len(state.plan.steps) if state.plan else 'No plan'}")
    
    context = runtime.context
    model = get_model(context.assessment_model or context.model)
    
    if not state.plan or state.step_index >= len(state.plan.steps):
        logger.warning("Invalid plan or step index - returning empty")
        return {}
    
    current_step = state.plan.steps[state.step_index]
    logger.info(f"Assessing step: {current_step.description}")
    logger.info(f"Tool: {current_step.tool_name}")
    
    # Extract the tool result from the last ToolMessage
    tool_result = None
    tool_name = current_step.tool_name
    
    for msg in reversed(state.messages[-10:]):
        if isinstance(msg, ToolMessage):
            try:
                content = get_message_text(msg)
                tool_result = json.loads(content) if content else {}
                logger.info(f"Found tool result success flag: {tool_result.get('success')}")
                break
            except json.JSONDecodeError:
                tool_result = {"message": content}
                logger.info(f"Non-JSON tool result: {content[:100]}")
                break
    
    if not tool_result:
        logger.warning("No tool result found - creating default assessment")
        assessment = AssessmentOutcome(
            outcome="retry",
            reason="No tool result found",
            confidence=0.5
        )
        return {
            "current_assessment": assessment,
            "total_assessments": state.total_assessments + 1
        }
    
    # Check if tool result indicates success
    tool_looks_successful = _is_tool_result_successful(tool_result, tool_name)
    logger.info(f"Tool appears successful: {tool_looks_successful}")
    
    # Create assessment based on tool result
    if tool_looks_successful:
        assessment = AssessmentOutcome(
            outcome="success",
            reason=f"Tool {tool_name} completed successfully with expected output",
            confidence=0.9
        )
        logger.info("Created SUCCESS assessment")
    else:
        # Check for explicit error in tool result
        tool_error = tool_result.get("error", "")
        if tool_error:
            assessment = AssessmentOutcome(
                outcome="retry",
                reason=f"Tool {tool_name} failed: {tool_error}",
                confidence=0.8
            )
            logger.info(f"Created RETRY assessment due to error: {tool_error}")
            
            # FIXED: Check if this should trigger error recovery
            if _should_trigger_error_recovery(tool_result, tool_error):
                logger.info("Triggering error recovery")
                error_context = {
                    "tool_result": tool_result,
                    "tool_name": tool_name,
                    "step_description": current_step.description,
                    "assessment": assessment,
                    "retry_count": state.retry_count
                }
                
                return {
                    "current_assessment": assessment,
                    "needs_error_recovery": True,
                    "error_context": error_context,
                    "total_assessments": state.total_assessments + 1
                    # Removed the hardcoded message - let error recovery provide the diagnosis
                }
        else:
            assessment = AssessmentOutcome(
                outcome="success",  # If no explicit error, assume success
                reason=f"Tool {tool_name} completed without explicit errors",
                confidence=0.7
            )
            logger.info("Created SUCCESS assessment (no explicit error)")
    
    # Prepare messages with narration
    messages_to_add = []
    
    # Handle completion or transitions
    if assessment.outcome == "success":
        next_index = state.step_index + 1
        is_final_step = next_index >= len(state.plan.steps)
        
        if is_final_step:
            completion_summary = await _generate_completion_summary(state, model, context)
            messages_to_add.append(AIMessage(content=completion_summary))
            logger.info("Added completion summary - this is the final step")
        elif next_index < len(state.plan.steps):
            next_step = state.plan.steps[next_index]
            transition = f"✅ Step completed successfully! Moving on to: {next_step.description}"
            messages_to_add.append(AIMessage(content=transition))
            logger.info("Added transition message")
    
    result = {
        "current_assessment": assessment,
        "total_assessments": state.total_assessments + 1
    }
    
    if messages_to_add:
        result["messages"] = messages_to_add
    
    logger.info(f"Final assessment outcome: {assessment.outcome}")
    logger.info(f"=== ASSESS DEBUG END ===")
    
    return result


def _should_trigger_error_recovery(tool_result: dict, error_message: str) -> bool:
    """Determine if this error should trigger error recovery."""
    if not tool_result.get("success", True):  # Explicit failure
        error_lower = error_message.lower()
        
        # Critical error patterns that need recovery
        critical_patterns = [
            "not found", "missing", "does not exist",
            "compilation failed", "build failed", 
            "dependency", "package", "import error",
            "configuration error", "invalid parameter",
            "permission denied", "access denied"
        ]
        
        if any(pattern in error_lower for pattern in critical_patterns):
            return True
    
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
                    if tool_name == "write_file":
                        recent_tool_results.append(f"Created file: {result.get('file_path', 'script file')}")
                    elif tool_name == "create_asset":
                        recent_tool_results.append(f"Created {result.get('asset_type', 'asset')}: {result.get('name', 'new asset')}")
                    elif tool_name == "compile_and_test":
                        errors = result.get('errors', 0)
                        recent_tool_results.append(f"Compilation: {errors} errors, {result.get('warnings', 0)} warnings")
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