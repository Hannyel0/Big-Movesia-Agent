"""Enhanced assessment node with intelligent error recovery integration."""

from __future__ import annotations

import json
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
from react_agent.graph.nodes.error_recovery import execute_error_recovery


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
        return bool(tool_result.get("snippets") or tool_result.get("available_snippets"))
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


def _is_critical_error(tool_result: dict, tool_name: str, assessment: AssessmentOutcome) -> bool:
    """Determine if this is a critical error that needs recovery."""
    if not tool_result:
        return False
    
    # Check for explicit failure
    if tool_result.get("success") is False:
        error_msg = tool_result.get("error", "").lower()
        
        # Critical error patterns
        critical_patterns = [
            "not found", "missing", "does not exist",
            "permission denied", "access denied",
            "compilation failed", "build failed",
            "dependency error", "reference error",
            "configuration error", "invalid parameter"
        ]
        
        if any(pattern in error_msg for pattern in critical_patterns):
            return True
    
    # Check assessment outcome
    if assessment and assessment.outcome in ["blocked", "retry"] and assessment.confidence > 0.7:
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


async def assess(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Enhanced assessment with intelligent error recovery integration."""
    context = runtime.context
    model = get_model(context.assessment_model or context.model)
    
    if not state.plan or state.step_index >= len(state.plan.steps):
        return {}
    
    current_step = state.plan.steps[state.step_index]
    
    # Extract the tool result from the last ToolMessage
    tool_result = None
    tool_name = current_step.tool_name
    
    for msg in reversed(state.messages[-10:]):
        if isinstance(msg, ToolMessage):
            try:
                content = get_message_text(msg)
                tool_result = json.loads(content) if content else {}
            except json.JSONDecodeError:
                tool_result = {"message": content}
            break
    
    # Check if tool result indicates success
    tool_looks_successful = _is_tool_result_successful(tool_result, tool_name)
    
    # Generate post-tool narration
    post_tool_narration = None
    if tool_result and tool_name:
        step_context = {
            "step_index": state.step_index,
            "total_steps": len(state.plan.steps),
            "goal": state.plan.goal,
            "tool_name": tool_name,
            "description": current_step.description,
            "retry_count": state.retry_count,
            "assessment": state.current_assessment
        }
        
        post_tool_narration = narration_engine.narrate_tool_result(
            tool_name,
            tool_result,
            step_context
        )
    
    # Enhanced assessment with error context
    retry_context = ""
    if state.retry_count > 0:
        retry_context = f"\n\nThis is retry attempt #{state.retry_count + 1}."
        if state.current_assessment:
            retry_context += f" Previous attempt failed because: {state.current_assessment.reason}"
    
    # Assess success with better error detection
    if tool_looks_successful and tool_name in ["create_asset", "get_script_snippets", "search", "get_project_info"]:
        assessment = AssessmentOutcome(
            outcome="success",
            reason=f"Tool {tool_name} completed successfully with expected output format",
            confidence=0.9
        )
    else:
        # Detailed LLM assessment
        assessment_request = f"""Assess if the current step has been successfully completed:

Overall Goal: {state.plan.goal}
Current Step ({state.step_index + 1}/{len(state.plan.steps)}): {current_step.description}
Success Criteria: {current_step.success_criteria}
Tool Used: {tool_name}{retry_context}

Tool Result: {json.dumps(tool_result, indent=2) if tool_result else "No result captured"}

ASSESSMENT CRITERIA:
- For simulated tools: success if they return expected data structure
- For errors: distinguish between retryable issues and systemic problems
- Consider if the error indicates missing dependencies, configuration issues, or tool malfunctions

Determine outcome (success/retry/blocked) and provide specific reasoning."""
        
        static_assessment_content = """You are evaluating game development step completion with focus on identifying recoverable errors.

Assessment Guidelines:
- SUCCESS: Tool returned expected data and format
- RETRY: Temporary issue, different parameters might work
- BLOCKED: Systemic issue requiring intervention (missing dependencies, configuration problems, etc.)

Be specific about WHY something failed - this helps with error recovery.
If blocked, clearly identify the type of problem (configuration, missing dependency, etc.)."""
        
        messages = [
            {"role": "system", "content": static_assessment_content},
            {"role": "user", "content": assessment_request}
        ]
        
        try:
            structured_model = model.with_structured_output(StructuredAssessment)
            structured_assessment = await structured_model.ainvoke(messages)
            
            assessment = AssessmentOutcome(
                outcome=structured_assessment.outcome,
                reason=structured_assessment.reason,
                fix=structured_assessment.fix or None,
                confidence=structured_assessment.confidence
            )
            
        except Exception as e:
            if tool_looks_successful:
                assessment = AssessmentOutcome(
                    outcome="success",
                    reason="Tool completed with expected output",
                    confidence=0.8
                )
            else:
                assessment = AssessmentOutcome(
                    outcome="retry",
                    reason=f"Assessment failed: {str(e)}",
                    confidence=0.5
                )
    
    # NEW: Check if this is a critical error that needs recovery
    if assessment.outcome != "success" and _is_critical_error(tool_result, tool_name, assessment):
        # Store error context for recovery
        error_context = {
            "tool_result": tool_result,
            "tool_name": tool_name,
            "step_description": current_step.description,
            "assessment": assessment,
            "retry_count": state.retry_count
        }
        
        # Trigger error recovery instead of normal retry/repair
        return {
            "current_assessment": assessment,
            "needs_error_recovery": True,
            "error_context": error_context,
            "messages": [AIMessage(content="I've detected an issue that needs systematic resolution. Let me analyze and fix this properly.")]
        }
    
    # Prepare messages with narration
    messages_to_add = []
    if post_tool_narration:
        messages_to_add.append(AIMessage(content=post_tool_narration))
    
    # Handle completion or transitions
    if assessment.outcome == "success":
        next_index = state.step_index + 1
        is_final_step = next_index >= len(state.plan.steps)
        
        if is_final_step:
            completion_summary = await _generate_completion_summary(state, model, context)
            messages_to_add.append(AIMessage(content=completion_summary))
        elif next_index < len(state.plan.steps):
            next_step = state.plan.steps[next_index]
            transition = f"✅ Step completed successfully! Moving on to: {next_step.description}"
            messages_to_add.append(AIMessage(content=transition))
    
    result = {
        "current_assessment": assessment,
        "total_assessments": state.total_assessments + 1
    }
    
    if messages_to_add:
        result["messages"] = messages_to_add
    
    # Complete streaming narration if active
    if context.runtime_metadata.get("supports_streaming"):
        streaming_narrator = StreamingNarrator()
        stream_id = f"step_{state.step_index}"
        if assessment.outcome == "success":
            streaming_narrator.complete_step(stream_id, "Step completed successfully!")
        elif assessment.outcome == "retry":
            streaming_narrator.update_step_progress(stream_id, "Retrying with adjustments...")
        else:
            streaming_narrator.complete_step(stream_id, "Step blocked - analyzing issue...")
    
    return result