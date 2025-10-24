"""Enhanced utility & helper functions for the ReAct agent."""


import hashlib
from typing import TYPE_CHECKING, Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

if TYPE_CHECKING:
    from react_agent.state import PlanStep


def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message, handling different content formats."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider:model' or 'provider/model'.

    Returns:
        BaseChatModel: Initialized chat model instance.
    """
    # Support both colon and slash formats for compatibility
    if ":" in fully_specified_name:
        provider, model = fully_specified_name.split(":", maxsplit=1)
    else:
        provider, model = fully_specified_name.split("/", maxsplit=1)

    # Enable stream_usage for proper token tracking in LangGraph Studio
    return init_chat_model(
        model,
        model_provider=provider,
        stream_usage=True
    )


def get_model(fully_specified_name: str) -> BaseChatModel:
    """Alias for load_chat_model for consistency with graph code.

    Args:
        fully_specified_name (str): String in the format 'provider:model' or 'provider/model'.

    Returns:
        BaseChatModel: Initialized chat model instance with token tracking enabled.

    Note:
        Token tracking is automatically enabled via stream_usage=True for LangGraph Studio.
        This ensures proper token usage display in the UI.
    """
    return load_chat_model(fully_specified_name)

def format_plan_visualization(plan: Any) -> str:
    """Create a visual representation of the execution plan.
    
    Args:
        plan: ExecutionPlan object
        
    Returns:
        String visualization of the plan
    """
    if not plan or not hasattr(plan, 'steps'):
        return "No plan available"
    
    lines = [f"🎯 Goal: {plan.goal}", ""]
    
    for i, step in enumerate(plan.steps):
        # Status indicator
        status_emoji = {
            "PENDING": "⏳",
            "IN_PROGRESS": "🔄",
            "SUCCEEDED": "✅",
            "FAILED": "❌",
            "BLOCKED": "🚫"
        }.get(step.status.value if hasattr(step.status, 'value') else str(step.status), "❓")
        
        # Step info
        lines.append(f"{status_emoji} Step {i+1}: {step.description}")
        
        # Tool info
        if step.tool_name:
            lines.append(f"   🔧 Tool: {step.tool_name}")
        
        # Success criteria
        lines.append(f"   📋 Success: {step.success_criteria}")
        
        # Dependencies
        if step.dependencies:
            lines.append(f"   🔗 Depends on: {step.dependencies}")
        
        # Attempts
        if step.attempts > 0:
            lines.append(f"   🔍 Attempts: {step.attempts}")
        
        # Errors
        if step.error_messages:
            lines.append(f"   ⚠️ Last error: {step.error_messages[-1][:100]}")
        
        lines.append("")
    
    return "\n".join(lines)


def create_dynamic_step_message(step: 'PlanStep', step_num: int, total_steps: int, goal: str) -> str:
    """Create varied, contextual step messages that feel more natural and agentic."""
    # Create deterministic but varied selection based on step content
    seed = f"{step.tool_name}_{step_num}_{step.description[:20]}"
    hash_val = int(hashlib.md5(seed.encode()).hexdigest(), 16) % 1000
    
    # Different message styles based on step position
    if step_num == 1:
        starters = [
            "Let me start by",
            "First, I'll",
            "To begin,", 
            "Starting with",
            "I'll kick things off by"
        ]
        starter = starters[hash_val % len(starters)]
    elif step_num == total_steps:
        starters = [
            "Finally, I'll",
            "To wrap this up, I'll",
            "Last step - I'll",
            "Completing this by",
            "To finish, I'll"
        ]
        starter = starters[hash_val % len(starters)]
    else:
        starters = [
            "Next, I'll",
            "Now I'll",
            "Moving on, I'll", 
            "Time to",
            "I'll proceed by",
            "Let me",
            "Going to"
        ]
        starter = starters[hash_val % len(starters)]
    
    # Tool-specific action phrases with variety
    tool_actions = {
        "search_project": [
            "search through your Unity project data",
            "query the indexed project database",
            "find assets and components in your project",
            "explore your project's structure and dependencies",
            "discover what's already in your Unity project"
        ],
        "code_snippets": [
            "find relevant code patterns using semantic search",
            "discover existing implementations in your scripts",
            "locate code that does what you need",
            "search through your C# scripts by functionality",
            "find the right code examples from your project"
        ],
        "file_operation": [
            "handle the file operations safely",
            "read, write, or modify project files",
            "manage your Unity project files",
            "perform secure file operations with validation",
            "work with your project's file system"
        ],
        "web_search": [
            "search for Unity documentation and tutorials",
            "find the latest game development best practices",
            "look up authoritative guidance on this topic",
            "research current Unity development approaches",
            "explore what the Unity community recommends"
        ]
    }
    
    # Get varied action for this tool
    actions = tool_actions.get(step.tool_name, ["work on this step"])
    action = actions[hash_val % len(actions)]
    
    # Add contextual details based on goal
    context_additions = []
    goal_lower = goal.lower()
    
    if "player" in goal_lower or "character" in goal_lower:
        context_additions = [
            "for your character system",
            "to get your player mechanics working",
            "so your character behaves correctly",
            ""
        ]
    elif "ui" in goal_lower or "menu" in goal_lower:
        context_additions = [
            "for your interface",
            "to get the UI working smoothly", 
            "for a polished user experience",
            ""
        ]
    elif "level" in goal_lower or "scene" in goal_lower:
        context_additions = [
            "for your level design",
            "to create the game environment",
            "for the scene layout",
            ""
        ]
    else:
        context_additions = ["", "for your game", "to make this work well", "properly"]
    
    context = context_additions[hash_val % len(context_additions)]
    
    # Add occasional progress indicators
    progress_phrases = []
    if step_num > 1:
        progress_phrases = [
            "",
            f"(step {step_num} of {total_steps})",
            f"- {step_num}/{total_steps}",
            ""
        ]
        progress = progress_phrases[hash_val % len(progress_phrases)]
    else:
        progress = ""
    
    # Combine everything naturally
    if context:
        message = f"{starter} {action} {context}"
    else:
        message = f"{starter} {action}"
    
    if progress:
        message = f"{message} {progress}"
    
    # Add some variety with dots vs no dots
    if hash_val % 3 == 0:
        message += "..."
    
    return message


def create_varied_post_tool_message(tool_name: str, result: dict, step_num: int) -> str:
    """Create varied post-tool messages instead of the repetitive ones."""
    # Create some variation based on step number and tool
    seed = f"{tool_name}_{step_num}_{str(result.get('success', True))}"
    hash_val = int(hashlib.md5(seed.encode()).hexdigest(), 16) % 100
    
    if not result.get("success", True):
        # Error responses
        error_messages = [
            "Hit a snag there, but I'll work around it.",
            "Encountered an issue - let me try a different approach.",
            "That didn't go as planned. Adjusting strategy.",
            "Running into some trouble, but I'll sort it out."
        ]
        return error_messages[hash_val % len(error_messages)]
    
    # Success responses by tool
    if tool_name == "search_project":
        count = result.get("result_count", 0)
        query = result.get("query_description", "query")
        responses = [
            f"Found {count} results for '{query}' in your project.",
            f"Located {count} matching items in the project database.",
            f"Project search returned {count} relevant results.",
            f"Discovered {count} items matching your criteria."
        ]
        return responses[hash_val % len(responses)]
    
    elif tool_name == "code_snippets":
        count = result.get("total_found", 0)
        query = result.get("query", "functionality")
        responses = [
            f"Found {count} code snippets for '{query}' functionality.",
            f"Located {count} relevant script examples.",
            f"Discovered {count} code patterns that match your needs.",
            f"Retrieved {count} useful implementations from your scripts."
        ]
        return responses[hash_val % len(responses)]
    
    elif tool_name == "file_operation":
        operation = result.get("operation", "operation")
        file_path = result.get("file_path", "file")
        if operation == "read":
            line_count = result.get("line_count", 0)
            responses = [
                f"Read {line_count} lines from {file_path}.",
                f"Successfully loaded content from {file_path}.",
                f"File {file_path} read successfully ({line_count} lines).",
                f"Retrieved content from {file_path}."
            ]
        elif operation == "write":
            line_count = result.get("line_count", 0)
            responses = [
                f"Wrote {line_count} lines to {file_path}.",
                f"Successfully created {file_path} with {line_count} lines.",
                f"File {file_path} written successfully.",
                f"Created {file_path} with your content."
            ]
        elif operation == "modify":
            modifications = result.get("modifications_applied", 0)
            responses = [
                f"Applied {modifications} modifications to {file_path}.",
                f"Successfully updated {file_path}.",
                f"File {file_path} modified successfully.",
                f"Changes applied to {file_path}."
            ]
        else:
            responses = [
                f"File operation '{operation}' completed on {file_path}.",
                f"Successfully performed {operation} on {file_path}.",
                f"{operation.title()} operation completed.",
                f"File {file_path} processed successfully."
            ]
        return responses[hash_val % len(responses)]
    
    elif tool_name == "web_search":
        count = result.get("result_count", 0)
        query = result.get("query", "topic")
        responses = [
            f"Found {count} web resources about '{query}'.",
            f"Located {count} helpful articles and tutorials.",
            f"Discovered {count} relevant Unity documentation pages.",
            f"Retrieved {count} useful resources from the web."
        ]
        return responses[hash_val % len(responses)]
    
    # Generic fallback
    generic_responses = [
        "Task completed successfully.",
        "Step finished.",
        "Operation complete.",
        "Done with this part.",
        "All done!"
    ]
    return generic_responses[hash_val % len(generic_responses)]