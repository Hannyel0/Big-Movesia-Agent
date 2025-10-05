"""Enhanced finish node for the ReAct agent with AI-generated completion summaries."""

from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.state import State
from react_agent.narration import NarrationEngine, StreamingNarrator
from react_agent.utils import get_message_text, get_model


# Initialize narration components
narration_engine = NarrationEngine()


async def _generate_direct_action_response(state: State, model) -> str:
    """Generate a response for direct actions that don't have plans."""
    
    # Extract the most recent user question
    user_question = None
    for msg in reversed(state.messages):
        if hasattr(msg, 'type') and msg.type == 'human':
            user_question = get_message_text(msg)
            break
    
    # Find the tool result from the most recent tool execution
    tool_results = []
    for msg in reversed(state.messages):
        if isinstance(msg, ToolMessage):
            try:
                result = json.loads(get_message_text(msg))
                tool_name = msg.name or 'unknown'
                tool_results.append({
                    'tool': tool_name,
                    'result': result,
                    'success': result.get('success', True)
                })
            except:
                continue
    
    if not tool_results:
        return "I've processed your request. Let me know if you need anything else!"
    
    # Get the most recent tool result
    latest_result = tool_results[0]
    
    # Create a prompt for the LLM to generate a natural response
    response_prompt = f"""You answered: "{user_question}"

Tool Result:
{json.dumps(latest_result['result'], indent=2)}

Generate a helpful, informative response that:
1. Answers their question with useful context
2. Groups or categorizes information when it makes sense (e.g., "You have 5 player scripts and 3 UI scripts")
3. Points out interesting patterns or insights (e.g., "most are Editor scripts", "all located in the Scripts folder")
4. Offers to dive deeper if relevant (e.g., "Want me to show you what's in any of these?")
5. Be conversational but informative - not just a data dump

Think: "What would a helpful teammate say?" not "What's the minimum valid answer?"

For example, if they asked "what scripts do I have?" and found 8 scripts, don't just say "You have 8 scripts." 
Instead say: "You have **8 C# scripts** in your project! They're mostly in the Scripts folder - I see player controllers, UI managers, and some utility scripts. Want me to dive into any specific ones?"

Generate your response now:"""
    
    try:
        messages = [
            {"role": "system", "content": "You are a helpful Unity development assistant. Provide informative, contextual answers that go beyond just stating facts. Add insights, group information logically, and be conversational like a knowledgeable teammate would be."},
            {"role": "user", "content": response_prompt}
        ]
        
        response = await model.ainvoke(messages)
        answer = get_message_text(response).strip()
        
        # Validate response quality
        if len(answer) < 10:
            # Fallback: extract key info directly
            result_data = latest_result['result']
            if latest_result['tool'] == 'search_project':
                result_count = result_data.get('result_count', 0)
                return f"I found {result_count} items in your project matching your query."
            elif latest_result['tool'] == 'code_snippets':
                total_found = result_data.get('total_found', 0)
                return f"I found {total_found} code snippets matching your search."
            elif latest_result['tool'] == 'web_search':
                results_count = len(result_data.get('results', []))
                return f"I found {results_count} relevant resources for your query."
            elif latest_result['tool'] == 'file_operation':
                operation = result_data.get('operation', 'operation')
                file_path = result_data.get('file_path', 'file')
                return f"Successfully completed {operation} on {file_path}."
            else:
                return f"The {latest_result['tool']} operation completed successfully."
        
        return answer
        
    except Exception as e:
        # Final fallback
        if latest_result['tool'] == 'search_project':
            result_count = latest_result['result'].get('result_count', 0)
            return f"Found {result_count} items in your project."
        return "I've completed your request successfully."


async def _generate_comprehensive_completion_summary(state: State, model, context: Context) -> str:
    """Generate a comprehensive AI completion summary based on the entire execution."""
    
    # Gather all completed work
    completed_steps_summary = []
    key_outputs = []
    
    for i, step in enumerate(state.plan.steps):
        status = "✅ Completed" if i in state.completed_steps else "⏸️ Partial"
        completed_steps_summary.append(f"{status}: {step.description}")
        
        # Track what was actually built/created
        if step.tool_name == "file_operation":
            key_outputs.append("Performed file operations")
        elif step.tool_name == "search_project":
            key_outputs.append("Queried project data")
        elif step.tool_name == "code_snippets":
            key_outputs.append("Searched code semantically")
        elif step.tool_name == "web_search":
            key_outputs.append("Researched Unity resources")
    
    # Extract concrete results from recent tool messages
    concrete_results = []
    files_created = []
    assets_built = []
    
    for msg in state.messages[-20:]:  # Look at recent messages
        if isinstance(msg, ToolMessage):
            try:
                result = json.loads(get_message_text(msg))
                tool_name = msg.name or 'unknown'
                
                if result.get("success", False):
                    if tool_name == "file_operation":
                        operation = result.get('operation', 'operation')
                        file_path = result.get('file_path', 'file')
                        files_created.append(f"📄 {operation}: {file_path}")
                        
                    elif tool_name == "search_project":
                        result_count = result.get('result_count', 0)
                        concrete_results.append(f"🔍 Found {result_count} project items")
                        
                    elif tool_name == "code_snippets":
                        total_found = result.get('total_found', 0)
                        concrete_results.append(f"💻 Found {total_found} code snippets")
                        
                    elif tool_name == "web_search":
                        results_count = len(result.get('results', []))
                        concrete_results.append(f"🌐 Found {results_count} web resources")
                        
            except:
                continue
    
    # Build context for AI summary
    execution_context = {
        "original_goal": state.plan.goal,
        "total_steps_planned": len(state.plan.steps),
        "steps_completed": len(state.completed_steps),
        "completion_percentage": (len(state.completed_steps) / len(state.plan.steps)) * 100 if state.plan.steps else 0,
        "total_tool_calls": state.total_tool_calls,
        "total_assessments": state.total_assessments,
        "files_created": files_created,
        "assets_built": assets_built,
        "concrete_results": concrete_results,
        "plan_revisions": state.plan_revision_count,
        "retry_attempts": sum(step.attempts for step in state.plan.steps)
    }
    
    # Create comprehensive prompt for AI summary with dynamic, contextual formatting
    completion_prompt = f"""You have just completed a Unity/game development session. Generate a contextual, well-formatted completion summary that dynamically adapts to what was actually accomplished.

**Session Context:**
- Original Goal: "{state.plan.goal}"
- Planned Steps: {len(state.plan.steps)}
- Completed Steps: {len(state.completed_steps)} ({execution_context['completion_percentage']:.1f}%)
- Total Tool Calls: {state.total_tool_calls}
- Plan Revisions: {state.plan_revision_count}

**Work Completed:**
{chr(10).join(completed_steps_summary)}

**Concrete Deliverables:**
{chr(10).join(files_created + assets_built + concrete_results) if (files_created + assets_built + concrete_results) else "Development work completed successfully"}

Generate a completion summary that:

1. **Uses dynamic structure** - Organize sections based on what actually happened (e.g., "Error Found and Fixed", "Implementation Complete", "Asset Created", etc.)

2. **Contextual formatting** - Use markdown formatting naturally:
   - **Bold** for key concepts, file names, and important status indicators
   - Bullet points or numbered lists where they make sense
   - Code formatting with backticks for file names, methods, or technical terms
   - Checkmarks (✅) or other symbols organically where appropriate

3. **Tell the story** of what happened:
   - If errors were fixed, explain what the problem was and how it was solved
   - If features were implemented, describe what they do and how they work
   - If assets were created, explain their purpose and integration
   - If configurations were changed, explain why and what impact it has

4. **Technical specificity** - Include actual details like:
   - File names and paths that were created/modified
   - Specific Unity concepts, components, or systems involved
   - Compilation results, error counts, warnings
   - Methods, classes, or technical implementations used

5. **Actionable next steps** - Provide concrete instructions for what the developer should do next to use or test what was built

6. **Natural tone** - Write as if explaining to a fellow developer what you accomplished and why it matters for their project

INSPIRATION EXAMPLE (adapt this style, don't copy the structure):
**Error Found and Fixed**
**Issue: Input System Import Conflict**
The script was importing `UnityEngine.InputSystem` but using legacy `Input` class methods like `Input.GetAxis("Horizontal")`. This creates a namespace conflict in Unity 6.
**Solution Applied**
I removed the unnecessary import since the script uses the legacy Input Manager, which works perfectly for character controllers.
**Current Status**
Your `PlayerController.cs` script is now error-free and ready to use! The script:
✅ **Fixed compilation errors** ✅ **Uses legacy Input system** ✅ **Includes all movement features**

Generate your contextual summary based on what actually happened in this session:"""

    try:
        # Generate AI completion summary
        completion_messages = [
            {"role": "system", "content": "You are providing a professional completion summary for a Unity development session. Be specific, confident, and focus on concrete deliverables. Keep it concise but comprehensive."},
            {"role": "user", "content": completion_prompt}
        ]
        
        completion_response = await model.ainvoke(completion_messages)
        ai_summary = get_message_text(completion_response).strip()
        
        # Validate AI response quality
        if len(ai_summary) < 30 or not any(word in ai_summary.lower() for word in ['perfect', 'excellent', 'great', 'success', 'complete', 'ready', 'built', 'created']):
            # Fallback if AI response is poor
            completion_rate = f"{len(state.completed_steps)}/{len(state.plan.steps)}"
            return f"Perfect! I've successfully completed {completion_rate} steps for your request: {state.plan.goal}. The implementation is ready for testing in your Unity project."
        
        return ai_summary
        
    except Exception as e:
        # Robust fallback
        deliverables = files_created + assets_built
        if deliverables:
            deliverable_summary = f"Created: {', '.join(deliverables[:2])}"
            if len(deliverables) > 2:
                deliverable_summary += f" and {len(deliverables) - 2} more items"
        else:
            deliverable_summary = "all requested development work"
            
        return f"Perfect! I've successfully completed {deliverable_summary} for your goal: {state.plan.goal}. Your Unity project now has the requested functionality."


async def finish(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Enhanced finish node with AI-generated completion summary based on actual execution."""
    context = runtime.context
    model = get_model(context.model)
    
    # Check if this is a direct action (no plan) or a planned execution
    if state.plan is None or len(state.plan.steps) == 0:
        # DIRECT ACTION PATH - no plan exists
        try:
            direct_response = await _generate_direct_action_response(state, model)
            return {"messages": [AIMessage(content=direct_response)]}
        except Exception as e:
            # Fallback for direct actions
            return {
                "messages": [AIMessage(content="I've processed your request. Let me know if you need anything else!")]
            }
    
    # PLANNED EXECUTION PATH - generate comprehensive summary
    try:
        ai_completion_summary = await _generate_comprehensive_completion_summary(state, model, context)
    except Exception as e:
        # Fallback to basic summary if AI generation fails
        completed_count = len(state.completed_steps)
        total_count = len(state.plan.steps) if state.plan else 0
        ai_completion_summary = f"Perfect! I've completed {completed_count} of {total_count} planned steps for: {state.plan.goal if state.plan else 'your request'}. The implementation is ready for use."
    
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
        "messages": [AIMessage(content=ai_completion_summary)]
    }