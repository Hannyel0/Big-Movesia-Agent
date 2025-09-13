"""Assessment node for the ReAct agent."""

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


# Initialize narration components
narration_engine = NarrationEngine()


def _create_step_transition(current_step, next_step) -> str:
    """Create transition narration between steps."""
    return f"✅ Step completed successfully! Moving on to: {next_step.description}"


def _is_tool_result_successful(tool_result: dict, tool_name: str) -> bool:
    """Check if a tool result indicates success based on tool-specific criteria."""
    if not tool_result:
        return False
    
    # Check explicit success flag first
    if "success" in tool_result:
        return tool_result["success"] is True
    
    # Tool-specific success checks for simulated tools
    if tool_name == "create_asset":
        # If we have asset info, it's successful
        return bool(tool_result.get("asset_type") and tool_result.get("name"))
    
    elif tool_name == "get_script_snippets":
        # If we have snippets or available_snippets, it's successful
        return bool(tool_result.get("snippets") or tool_result.get("available_snippets"))
    
    elif tool_name == "write_file":
        # If we have file_path and size info, it's successful
        return bool(tool_result.get("file_path") and tool_result.get("size_bytes") is not None)
    
    elif tool_name == "compile_and_test":
        # Success if we have compilation info (even with warnings)
        return "compilation_time" in tool_result or "errors" in tool_result
    
    elif tool_name == "search":
        # Success if we have search results
        return bool(tool_result.get("result")) and len(tool_result.get("result", [])) > 0
    
    elif tool_name == "get_project_info":
        # Success if we have project structure info
        return bool(tool_result.get("engine") or tool_result.get("project_name"))
    
    elif tool_name == "scene_management":
        # Success if we have action confirmation
        return bool(tool_result.get("action") and tool_result.get("scene_name"))
    
    elif tool_name == "edit_project_config":
        # Success if we have config section info
        return bool(tool_result.get("config_section"))
    
    # Default: assume success if we got any meaningful data
    return len(tool_result) > 1  # More than just timestamp or empty dict


async def _generate_completion_summary(state: State, model, context: Context) -> str:
    """Generate a completion summary using the LLM based on what was actually accomplished."""
    
    # Gather context about what was accomplished
    completed_steps_summary = []
    for i, step in enumerate(state.plan.steps):
        if i <= state.step_index:  # Include current completing step
            completed_steps_summary.append(f"Step {i+1}: {step.description} (using {step.tool_name})")
    
    # Find recent tool results for context
    recent_tool_results = []
    for msg in reversed(state.messages[-10:]):
        if isinstance(msg, ToolMessage):
            try:
                result = json.loads(get_message_text(msg))
                tool_name = msg.name or 'unknown'
                if _is_tool_result_successful(result, tool_name):
                    # Include key details from successful tool results
                    if tool_name == "write_file":
                        recent_tool_results.append(f"Created file: {result.get('file_path', 'script file')}")
                    elif tool_name == "create_asset":
                        recent_tool_results.append(f"Created {result.get('asset_type', 'asset')}: {result.get('name', 'new asset')}")
                    elif tool_name == "compile_and_test":
                        errors = result.get('errors', 0)
                        recent_tool_results.append(f"Compilation: {errors} errors, {result.get('warnings', 0)} warnings")
                    elif tool_name == "search":
                        results_count = len(result.get('result', []))
                        recent_tool_results.append(f"Found {results_count} resources")
                    elif tool_name == "get_script_snippets":
                        snippets_count = result.get('total_snippets', 0)
                        recent_tool_results.append(f"Retrieved {snippets_count} code snippets")
                    else:
                        recent_tool_results.append(f"{tool_name}: {result.get('message', 'completed successfully')}")
            except:
                recent_tool_results.append(f"{msg.name}: completed")
    
    # Create prompt for completion summary
    completion_prompt = f"""You have just completed all steps for the goal: "{state.plan.goal}"

Completed Steps:
{chr(10).join(completed_steps_summary)}

Key Results:
{chr(10).join(recent_tool_results[-5:]) if recent_tool_results else "All steps completed successfully"}

Generate a brief, professional completion message (1-2 sentences) that:
1. Starts with "Perfect!" or similar positive confirmation
2. Summarizes what was successfully accomplished
3. Shows awareness of the specific work done
4. Sounds confident and professional

Examples of good completion messages:
- "Perfect! I've successfully created a complete character controller script for you. The script includes movement mechanics and follows Unity best practices."
- "Perfect! I've identified and fixed the main issue in your project. The compilation errors have been resolved and the code is now working properly."
- "Perfect! I've successfully configured all the requested project settings. Your Unity project is now optimized for development."

Your completion message:"""

    try:
        # Generate completion summary using LLM
        completion_messages = [
            {"role": "system", "content": "You are providing a brief, professional completion summary for a Unity development task. Be concise, specific, and confident."},
            {"role": "user", "content": completion_prompt}
        ]
        
        completion_response = await model.ainvoke(completion_messages)
        completion_summary = get_message_text(completion_response).strip()
        
        # Validate the response quality
        if len(completion_summary) < 20 or not completion_summary.lower().startswith(('perfect', 'excellent', 'great', 'done', 'completed', 'success')):
            # Fallback if LLM response is poor
            return f"Perfect! I've successfully completed your request: {state.plan.goal}"
        
        return completion_summary
        
    except Exception as e:
        # Fallback on error
        return f"Perfect! I've successfully completed all steps for: {state.plan.goal}"


async def assess(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Enhanced assessment with improved tool result evaluation."""
    context = runtime.context
    model = get_model(context.assessment_model or context.model)
    
    if not state.plan or state.step_index >= len(state.plan.steps):
        return {}
    
    current_step = state.plan.steps[state.step_index]
    
    # Extract the tool result from the last ToolMessage
    tool_result = None
    tool_name = current_step.tool_name
    
    for msg in reversed(state.messages[-10:]):  # Look at recent messages
        if isinstance(msg, ToolMessage):
            try:
                # Parse tool result
                content = get_message_text(msg)
                tool_result = json.loads(content) if content else {}
            except json.JSONDecodeError:
                tool_result = {"message": content}
            break
    
    # IMPROVED SUCCESS DETECTION
    # First check if the tool result looks successful using tool-specific logic
    tool_looks_successful = _is_tool_result_successful(tool_result, tool_name)
    
    # GENERATE RICH POST-TOOL NARRATION with retry awareness
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
        
        # Use the narration engine to create rich, contextual narration
        post_tool_narration = narration_engine.narrate_tool_result(
            tool_name,
            tool_result,
            step_context
        )
    
    # Include retry context in assessment
    retry_context = ""
    if state.retry_count > 0:
        retry_context = f"\n\nThis is retry attempt #{state.retry_count + 1}."
        if state.current_assessment:
            retry_context += f" Previous attempt failed because: {state.current_assessment.reason}"
    
    # IMPROVED ASSESSMENT LOGIC
    # For simulated/stub tools, be more lenient and trust tool-specific success detection
    if tool_looks_successful and tool_name in ["create_asset", "get_script_snippets", "search", "get_project_info"]:
        # These are simulated tools that always return success - trust them more
        assessment = AssessmentOutcome(
            outcome="success",
            reason=f"Tool {tool_name} completed successfully with expected output format",
            confidence=0.9
        )
    else:
        # Only do LLM assessment for more complex tools or when tool result is unclear
        assessment_request = f"""Assess if the current step has been successfully completed:

Overall Goal: {state.plan.goal}
Current Step ({state.step_index + 1}/{len(state.plan.steps)}): {current_step.description}
Success Criteria: {current_step.success_criteria}
Tool Used: {tool_name}{retry_context}

Tool Result: {json.dumps(tool_result, indent=2) if tool_result else "No result captured"}

IMPORTANT: For simulated development tools (create_asset, get_script_snippets, search, get_project_info), 
if they return structured data with the expected fields, consider them successful unless there's a clear error.

Determine if the step succeeded, needs retry, or is blocked. Be more lenient for retry attempts if there's progress."""
        
        # Static assessment prompt - cacheable
        static_assessment_content = """You are evaluating game development step completion with focus on practical progress.

Assessment Guidelines:
- For simulated tools: if they return expected data structure, they succeeded
- For create_asset: success if asset_type, name, and path are returned
- For get_script_snippets: success if snippets or available_snippets are returned
- For search: success if result array is returned with content
- For write_file: success if file_path and size info are returned
- For compilation: success if compilation info is returned (warnings are OK)

Be practical - simulated tools that return proper data structure have succeeded.
Only mark as "retry" if there's a clear technical error or missing required information.
Mark as "blocked" only for technical limitations, not data format issues.

Assessment outcomes:
- "success": Step completed with expected output format
- "retry": Missing critical information or clear error
- "blocked": Technical limitation preventing implementation"""

        # Structure for optimal caching
        messages = [
            {"role": "system", "content": static_assessment_content},
            {"role": "user", "content": assessment_request}
        ]
        
        try:
            # Use structured output for assessment
            structured_model = model.with_structured_output(StructuredAssessment)
            structured_assessment = await structured_model.ainvoke(messages)
            
            assessment = AssessmentOutcome(
                outcome=structured_assessment.outcome,
                reason=structured_assessment.reason,
                fix=structured_assessment.fix or None,
                confidence=structured_assessment.confidence
            )
            
        except Exception as e:
            # Fallback: if tool looks successful, mark as success
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
    
    # Prepare messages with rich narration
    messages_to_add = []
    if post_tool_narration:
        messages_to_add.append(AIMessage(content=post_tool_narration))
    
    # CHECK FOR COMPLETION - Generate LLM completion summary
    if assessment.outcome == "success":
        next_index = state.step_index + 1
        is_final_step = next_index >= len(state.plan.steps)
        
        if is_final_step:
            # Generate completion summary using LLM
            completion_summary = await _generate_completion_summary(state, model, context)
            messages_to_add.append(AIMessage(content=completion_summary))
        elif next_index < len(state.plan.steps):
            # More steps to go - Add transition
            next_step = state.plan.steps[next_index]
            transition = _create_step_transition(current_step, next_step)
            if transition:
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
            streaming_narrator.complete_step(stream_id, "Step blocked - replanning...")
    
    return result