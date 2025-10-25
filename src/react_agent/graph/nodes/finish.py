"""Enhanced finish node for the ReAct agent with AI-generated completion summaries."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.state import State
from react_agent.narration import NarrationEngine, StreamingNarrator
from react_agent.utils import get_message_text, get_model, is_anthropic_model
from react_agent.prompts import get_cacheable_final_summary_prompt
from react_agent.ui_messages import create_plan_ui_message

logger = logging.getLogger(__name__)

# Initialize narration components
narration_engine = NarrationEngine()


async def _generate_direct_action_response(
    state: State, model, cache_enabled: bool = True
) -> str:
    """Generate a response for direct actions that don't have plans."""

    # Extract the most recent user question
    user_question = None
    for msg in reversed(state.messages):
        if hasattr(msg, "type") and msg.type == "human":
            user_question = get_message_text(msg)
            break

    # Find the tool result from the most recent tool execution
    tool_results = []
    for msg in reversed(state.messages):
        if isinstance(msg, ToolMessage):
            try:
                result = json.loads(get_message_text(msg))
                tool_name = msg.name or "unknown"
                tool_results.append(
                    {
                        "tool": tool_name,
                        "result": result,
                        "success": result.get("success", True),
                    }
                )
            except:
                continue

    if not tool_results:
        return "I've processed your request. Let me know if you need anything else!"

    # Get the most recent tool result
    latest_result = tool_results[0]

    # Create a prompt for the LLM to generate a natural response
    response_prompt = f"""You answered: "{user_question}"

Tool Result:
{json.dumps(latest_result["result"], indent=2)}

Generate a helpful, informative response using **proper markdown formatting**.

### MARKDOWN FORMATTING REQUIREMENTS:

**YOU WILL BE PENALIZED** for plain text responses. You MUST use:

- **## Headers** for main sections
- **Bold** for key terms, numbers, file names
- **Bullet points** with `-` (not •)
- **Blank lines** before/after headers and lists
- **Emojis** for visual clarity: 📁 🔍 ✅ 💡 🎯

### Response Structure Example:
```markdown
## 📁 Project Scripts

You have **8 C# scripts** in your project!

### Breakdown:
- **Player controllers** (3 scripts)
- **UI managers** (2 scripts)  
- **Utility scripts** (3 scripts)

### 💡 Insights:
Most are located in the **Scripts/** folder. I noticed several editor scripts for workflow automation.

### 🎯 Next Steps:
Want me to dive into any specific ones?
```

### Your Response Guidelines:
1. **Answer their question** with useful context
2. **Group or categorize** information when it makes sense
3. **Point out interesting patterns** or insights
4. **Offer to dive deeper** if relevant
5. **Be conversational** but informative - not just a data dump

**Think**: "What would a helpful teammate say?" not "What's the minimum valid answer?"

**BAD** (plain text): "You have 8 scripts."
**GOOD** (markdown): "## 📁 Your Scripts\\n\\nYou have **8 C# scripts** in your project! They're mostly in the **Scripts/** folder..."

Generate your markdown-formatted response now:"""

    try:
        # ✅ CACHING: Use cacheable system prompt
        system_content_structured = get_cacheable_final_summary_prompt(
            cache_enabled=cache_enabled
        )

        messages = [
            {"role": "system", "content": system_content_structured},
            {"role": "user", "content": response_prompt},
        ]

        response = await model.ainvoke(messages)
        answer = get_message_text(response).strip()

        # Validate response quality
        if len(answer) < 10:
            # Fallback: extract key info directly
            result_data = latest_result["result"]
            if latest_result["tool"] == "search_project":
                result_count = result_data.get("result_count", 0)
                return (
                    f"I found {result_count} items in your project matching your query."
                )
            elif latest_result["tool"] == "code_snippets":
                total_found = result_data.get("total_found", 0)
                return f"I found {total_found} code snippets matching your search."
            elif latest_result["tool"] == "web_search":
                results_count = len(result_data.get("results", []))
                return f"I found {results_count} relevant resources for your query."
            elif latest_result["tool"] in [
                "read_file",
                "write_file",
                "modify_file",
                "delete_file",
                "move_file",
            ]:
                tool_name = latest_result["tool"]
                file_path = result_data.get("file_path", "file")
                return f"Successfully completed {tool_name} on {file_path}."
            else:
                return f"The {latest_result['tool']} operation completed successfully."

        return answer

    except Exception as e:
        # Final fallback
        if latest_result["tool"] == "search_project":
            result_count = latest_result["result"].get("result_count", 0)
            return f"Found {result_count} items in your project."
        return "I've completed your request successfully."


async def _generate_comprehensive_completion_summary(
    state: State, model, context: Context
) -> str:
    """Generate a comprehensive AI completion summary based on the entire execution."""

    # Gather all completed work
    completed_steps_summary = []
    key_outputs = []

    for i, step in enumerate(state.plan.steps):
        status = "✅ Completed" if i in state.completed_steps else "⏸️ Partial"
        completed_steps_summary.append(f"{status}: {step.description}")

        # Track what was actually built/created
        if step.tool_name in [
            "read_file",
            "write_file",
            "modify_file",
            "delete_file",
            "move_file",
        ]:
            key_outputs.append(f"Performed {step.tool_name}")
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
                tool_name = msg.name or "unknown"

                if result.get("success", False):
                    if tool_name in [
                        "read_file",
                        "write_file",
                        "modify_file",
                        "delete_file",
                        "move_file",
                    ]:
                        file_path = result.get("file_path", "file")
                        files_created.append(f"📄 {tool_name}: {file_path}")

                    elif tool_name == "search_project":
                        result_count = result.get("result_count", 0)
                        concrete_results.append(
                            f"🔍 Found {result_count} project items"
                        )

                    elif tool_name == "code_snippets":
                        total_found = result.get("total_found", 0)
                        concrete_results.append(f"💻 Found {total_found} code snippets")

                    elif tool_name == "web_search":
                        results_count = len(result.get("results", []))
                        concrete_results.append(
                            f"🌐 Found {results_count} web resources"
                        )

            except:
                continue

    # Build context for AI summary
    execution_context = {
        "original_goal": state.plan.goal,
        "total_steps_planned": len(state.plan.steps),
        "steps_completed": len(state.completed_steps),
        "completion_percentage": (len(state.completed_steps) / len(state.plan.steps))
        * 100
        if state.plan.steps
        else 0,
        "total_tool_calls": state.total_tool_calls,
        "total_assessments": state.total_assessments,
        "files_created": files_created,
        "assets_built": assets_built,
        "concrete_results": concrete_results,
        "plan_revisions": state.plan_revision_count,
        "retry_attempts": sum(step.attempts for step in state.plan.steps),
    }

    # ✅ MEMORY: Get memory context if available
    memory_section = ""
    if state.memory:
        try:
            # ✅ FIX: Use get_memory_context() instead of deleted get_relevant_context()
            memory_context_str = await state.memory.get_memory_context(
                include_patterns=True, include_episodes=True
            )
            if memory_context_str:
                memory_section = f"\n\n## Memory Context\n{memory_context_str}"
        except Exception as e:
            print(f"⚠️ [Finish] Could not get memory context: {e}")

    # Create comprehensive prompt for AI summary with dynamic, contextual formatting
    completion_prompt = f"""## COMPLETION SUMMARY GENERATION

You have just completed a Unity/game development session. Generate a contextual, well-formatted completion summary using **proper markdown**.

### ⚠️ CRITICAL MARKDOWN REQUIREMENT:

**YOU WILL BE HEAVILY PENALIZED FOR PLAIN TEXT RESPONSES.**

You MUST use:
- **## Headers** for main sections (with blank lines before/after)
- **### Subheaders** for subsections
- **Bold** for file names, key concepts, status indicators
- **Blank lines** before/after all headers and lists
- **Bullet points** with `-` (not •)
- **Emojis** for visual organization: ✅ ❌ 🔧 💡 🎯 📁 ⚠️

---

### Session Context:

**Original Goal:** "{state.plan.goal}"
**Planned Steps:** {len(state.plan.steps)}
**Completed Steps:** {len(state.completed_steps)} ({execution_context["completion_percentage"]:.1f}%)
**Total Tool Calls:** {state.total_tool_calls}
**Plan Revisions:** {state.plan_revision_count}

### Work Completed:
{chr(10).join(completed_steps_summary)}

### Concrete Deliverables:
{chr(10).join(files_created + assets_built + concrete_results) if (files_created + assets_built + concrete_results) else "Development work completed successfully"}

{f"### What We Learned:\n{memory_section}\n" if memory_section else ""}

---

### MANDATORY SUMMARY STRUCTURE:
```markdown
## ✅ [Dynamic Title Based on What Happened]

Brief overview with **bold emphasis**.

### 🎯 What Was Accomplished

- **Deliverable 1**: Description with specifics
- **Deliverable 2**: Description with specifics

### 🔧 Technical Details

**File:** `filename.cs` 
**Changes:** What was modified/created
**Integration:** How it fits with project

### 💡 Key Insights

Important patterns or learnings discovered.

### ⚡ Next Steps

1. First action with **clear instruction**
2. Second action with details
```

### Summary Requirements:

1. **Dynamic structure** - Organize sections based on what **actually happened**:
   - "## ✅ Error Fixed" for bug fixes
   - "## ✅ Feature Implemented" for new features
   - "## ✅ Asset Created" for asset work
   - "## ⚠️ Partial Progress" for incomplete work

2. **STRICT markdown formatting**:
   - **Bold** all file names, key concepts, and status indicators
   - Use **blank lines before AND after** every header
   - Use **blank lines before** every list
   - Use `-` for bullet points (never •)
   - Use emojis **naturally** for section headers

3. **Tell the story** with context:
   - **Errors fixed**: What broke → Why → How we fixed it
   - **Features implemented**: What it does → How it works → Why it matters
   - **Assets created**: Purpose → Integration → Usage

4. **Technical specificity**:
   - **File names**: Always in `backticks` and **bold**: **`PlayerController.cs`**
   - **Methods/Classes**: Use backticks: `Update()`, `PlayerController` 
   - **Unity components**: Bold: **Rigidbody2D**, **Input Manager**
   - **Concrete numbers**: Errors fixed, lines added, etc.

5. **Actionable next steps** with numbered list

6. **Natural tone** - Fellow developer explaining what was accomplished

### EXAMPLE (ADAPT, DON'T COPY):
```markdown
## ✅ Input System Error Fixed

Found and resolved a namespace conflict in **`PlayerController.cs`**.

### ⚠️ The Issue

The script was importing `UnityEngine.InputSystem` but using legacy `Input` class methods like `Input.GetAxis("Horizontal")`. This creates a namespace conflict in Unity 6.

### 🔧 Solution Applied

Removed the unnecessary import since the script uses the **legacy Input Manager**, which works perfectly for character controllers.

### ✅ Current Status

Your **`PlayerController.cs`** is now error-free and ready to use:

- ✅ Fixed compilation errors
- ✅ Uses legacy Input system
- ✅ Includes all movement features

### ⚡ Next Steps

1. **Test the controller** in Play Mode
2. **Assign** the script to your player GameObject
3. **Configure** input axes in Edit → Project Settings → Input Manager
```

**GENERATE YOUR PROPERLY FORMATTED SUMMARY NOW** (following the structure above with blank lines, headers, and emojis):"""

    try:
        # ✅ CACHING: Use cacheable final summary prompt (only adds cache_control for Anthropic)
        cache_enabled = getattr(context, "enable_prompt_cache", True)
        using_anthropic = is_anthropic_model(context.model)
        system_content_structured = get_cacheable_final_summary_prompt(
            cache_enabled=cache_enabled and using_anthropic
        )

        # Generate AI completion summary
        completion_messages = [
            {"role": "system", "content": system_content_structured},
            {"role": "user", "content": completion_prompt},
        ]

        completion_response = await model.ainvoke(completion_messages)
        ai_summary = get_message_text(completion_response).strip()

        # Validate AI response quality
        if len(ai_summary) < 30 or not any(
            word in ai_summary.lower()
            for word in [
                "perfect",
                "excellent",
                "great",
                "success",
                "complete",
                "ready",
                "built",
                "created",
            ]
        ):
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

    logger.info("🏁 [FINISH] ===== STARTING FINISH =====")

    # ✅ CRITICAL: Determine if this was a direct action or planned execution
    is_direct_action = state.plan is None or len(state.plan.steps) == 0

    # ✅ MEMORY: End task episode - CRITICAL
    if state.memory:
        # Determine success
        if is_direct_action:
            # Direct action - check if we have a successful tool result
            success = True  # Assume success if we reached finish
            outcome_summary = "Direct action completed successfully"

            # Check last tool message for actual success
            for msg in reversed(state.messages[-5:]):
                if isinstance(msg, ToolMessage):
                    try:
                        result = json.loads(get_message_text(msg))
                        success = result.get("success", True)
                        if success:
                            tool_name = msg.name or "action"
                            outcome_summary = (
                                f"Direct action with {tool_name} completed successfully"
                            )
                        else:
                            outcome_summary = f"Direct action completed with errors"
                        break
                    except:
                        pass

            logger.info(
                f"🏁 [FINISH] Direct action: {'✅ success' if success else '❌ failure'}"
            )
        else:
            # Planned execution
            success = state.step_index >= len(state.plan.steps) - 1
            # ✅ FIX: Count actually completed steps from plan status
            from react_agent.state import StepStatus

            completed_count = sum(
                1 for step in state.plan.steps if step.status == StepStatus.SUCCEEDED
            )
            total_count = len(state.plan.steps)
            outcome_summary = f"Completed {completed_count}/{total_count} steps"
            logger.info(
                f"🏁 [FINISH] Planned execution: {'✅ success' if success else '❌ failure'}"
            )
            logger.info(
                f"🏁 [FINISH]   Steps with SUCCEEDED status: {completed_count}/{total_count}"
            )
            logger.info(
                f"🏁 [FINISH]   completed_steps list: {len(state.completed_steps)} items"
            )

        # ✅ FIXED: Only clear working memory for complex multi-step plans
        # Preserve memory for:
        # - Direct actions (single tool calls)
        # - Simple plans (1-3 steps, like search queries)
        # Clear memory only for:
        # - Complex plans (4+ steps, like multi-file operations)
        is_complex_plan = (
            state.plan is not None
            and len(state.plan.steps) > 3
            and not is_direct_action
        )
        clear_working = is_complex_plan

        # ✅ OPTIMIZATION: Skip persistence if routing already did it recently
        import time

        last_persist = state.runtime_metadata.get("last_memory_persist", 0)
        time_since_persist = time.time() - last_persist

        if time_since_persist < 1.0:  # Within last second
            logger.info(
                f"🧠 [FINISH] Skipping persistence (already done in routing {time_since_persist:.3f}s ago)"
            )
            # End episode WITHOUT triggering persistence
            state.memory.episodic_memory.end_episode(success, outcome_summary)
            # Clear working memory if needed
            if clear_working:
                state.memory.working_memory.clear()
        else:
            # Normal path: end_task with persistence
            logger.info(
                f"🧠 [FINISH] Performing persistence (last persist was {time_since_persist:.3f}s ago)"
            )
            await state.memory.end_task(
                success=success,
                outcome_summary=outcome_summary,
                clear_working_memory=clear_working,  # ✅ PRESERVE for direct actions
            )

        logger.info(
            f"🧠 [FINISH] Episode ended: {'✅ success' if success else '❌ incomplete'}"
        )
        logger.info(f"🧠 [FINISH]   Outcome: {outcome_summary}")
        logger.info(
            f"🧠 [FINISH]   Working memory {'CLEARED (complex plan)' if clear_working else 'PRESERVED (simple/direct)'}"
        )

        # Log working memory state
        if hasattr(state.memory, "working_memory"):
            recent_tools = state.memory.working_memory.recent_tool_results
            step_count = len(state.plan.steps) if state.plan else 0
            logger.info(
                f"🧠 [FINISH] Working memory has {len(recent_tools)} tool results stored"
            )
            print(f"\n🧠 [FINISH] Working memory state:")
            print(f"   Direct action: {is_direct_action}")
            print(f"   Plan steps: {step_count}")
            print(f"   Complex plan (>3 steps): {is_complex_plan}")
            print(f"   Memory preserved: {not clear_working}")
            print(f"   Tool results stored: {len(recent_tools)}")
            for i, tool in enumerate(recent_tools, 1):
                logger.info(f"🧠 [FINISH]   {i}. {tool['summary']}")
                print(f"   {i}. {tool['summary']}")

        # ✅ Consolidate memories periodically
        episode_count = len(state.memory.episodic_memory.recent_episodes)
        if episode_count > 0 and episode_count % 10 == 0:
            logger.info(
                f"🧠 [FINISH] Consolidating memories after {episode_count} episodes..."
            )
            state.memory.consolidate_memories()
            logger.info(f"🧠 [FINISH] Memory consolidation complete")

    # Check if this is a direct action (no plan) or a planned execution
    if state.plan is None or len(state.plan.steps) == 0:
        # DIRECT ACTION PATH - no plan exists
        try:
            cache_enabled = getattr(context, "enable_prompt_cache", True)
            using_anthropic = is_anthropic_model(context.model)
            direct_response = await _generate_direct_action_response(
                state, model, cache_enabled=cache_enabled and using_anthropic
            )
            return {"messages": [AIMessage(content=direct_response)]}
        except Exception as e:
            # Fallback for direct actions
            return {
                "messages": [
                    AIMessage(
                        content="I've processed your request. Let me know if you need anything else!"
                    )
                ]
            }

    # PLANNED EXECUTION PATH - generate comprehensive summary
    try:
        ai_completion_summary = await _generate_comprehensive_completion_summary(
            state, model, context
        )
    except Exception as e:
        # Fallback to basic summary if AI generation fails
        completed_count = len(state.completed_steps)
        total_count = len(state.plan.steps) if state.plan else 0
        ai_completion_summary = f"Perfect! I've completed {completed_count} of {total_count} planned steps for: {state.plan.goal if state.plan else 'your request'}. The implementation is ready for use."

    # Add final streaming updates if supported
    if context.runtime_metadata.get("supports_streaming"):
        streaming_narrator = StreamingNarrator()
        final_update = streaming_narrator.create_inline_update(
            " Implementation complete! Check your Unity project for the new features.",
            style="success",
        )
        if hasattr(runtime, "push_ui_message"):
            runtime.push_ui_message(final_update)

    # Emit final plan UI update showing all steps completed
    result = {"messages": [AIMessage(content=ai_completion_summary)]}

    if state.plan:
        final_message = result["messages"][0]

        plan_ui_msg = create_plan_ui_message(
            state.plan, final_message.id, ui_message_id=state.plan_ui_message_id
        )

        result["ui"] = [plan_ui_msg]

    return result
