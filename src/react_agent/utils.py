"""Enhanced utility & helper functions for the ReAct agent."""


import hashlib
from typing import Any, TYPE_CHECKING

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
        fully_specified_name (str): String in the format 'provider/model'.
        
    Returns:
        BaseChatModel: Initialized chat model instance.
    """
    provider, model = fully_specified_name.split("/", maxsplit=1)
    return init_chat_model(model, model_provider=provider)


def get_model(fully_specified_name: str) -> BaseChatModel:
    """Alias for load_chat_model for consistency with graph code.
    
    Args:
        fully_specified_name (str): String in the format 'provider/model'.
        
    Returns:
        BaseChatModel: Initialized chat model instance.
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
        "search": [
            "dig into the latest Unity best practices",
            "research current game development approaches", 
            "look up authoritative guidance on this",
            "find the most recent tutorials and docs",
            "explore what the Unity community recommends"
        ],
        "get_project_info": [
            "examine your project setup carefully",
            "inspect the current Unity configuration",
            "analyze your project structure and settings",
            "check what you're working with here",
            "understand your development environment"
        ],
        "get_script_snippets": [
            "grab some proven code patterns for this",
            "fetch battle-tested script templates",
            "pull up the right code examples",
            "get some solid implementation patterns",
            "find the perfect code snippets for your needs"
        ],
        "write_file": [
            "create that script file for you",
            "write the implementation to your project",
            "build the actual code file",
            "craft the script you need",
            "put together the working code"
        ],
        "compile_and_test": [
            "run a quick build check",
            "make sure everything compiles cleanly", 
            "verify this integrates properly",
            "test that nothing breaks",
            "ensure the code works as expected"
        ],
        "scene_management": [
            "set up your scene properly",
            "configure the scene layout",
            "organize your scene structure",
            "prepare the scene environment",
            "handle the scene setup"
        ],
        "edit_project_config": [
            "adjust your project settings",
            "fine-tune the configuration",
            "update the project parameters",
            "modify the settings safely",
            "configure this properly"
        ],
        "create_asset": [
            "build that asset for your project",
            "create the game asset you need",
            "construct the new project element",
            "generate the required asset",
            "make the asset with proper setup"
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
    if tool_name == "search":
        n_results = len(result.get("result", []) or [])
        responses = [
            f"Found {n_results} helpful sources - using the best ones.",
            f"Great! Discovered {n_results} relevant resources.",
            f"Perfect - got {n_results} quality references to work with.",
            f"Excellent findings - {n_results} sources that should help."
        ]
        return responses[hash_val % len(responses)]
    
    elif tool_name == "write_file":
        lines = result.get('lines_written', 0)
        path = result.get('file_path', 'project')
        responses = [
            f"Script created! Wrote {lines} lines to {path}.",
            f"File ready - {lines} lines of code in {path}.",
            f"Done! Created {lines} lines in {path}.",
            f"Script complete - {lines} lines saved to {path}."
        ]
        return responses[hash_val % len(responses)]
    
    elif tool_name == "compile_and_test":
        if result.get("errors", 0) == 0:
            responses = [
                "Build successful! No errors detected.",
                "Clean compile - everything looks good.",
                "Perfect! Code builds without issues.",
                "All systems go - no compilation problems."
            ]
        else:
            responses = [
                "Found some issues that need fixing.",
                "Compilation revealed a few problems to address.",
                "Build flagged some errors - I'll help resolve them.",
                "Some compiler feedback to work through."
            ]
        return responses[hash_val % len(responses)]
    
    elif tool_name == "get_project_info":
        engine = result.get("engine", "Unity")
        version = result.get("version", "")
        responses = [
            f"Got the full picture of your {engine} {version} setup.",
            f"Project analysis complete - {engine} {version} environment mapped.",
            f"Perfect! Understanding your {engine} project structure now.",
            f"Project details captured - working with {engine} {version}."
        ]
        return responses[hash_val % len(responses)]
    
    elif tool_name == "get_script_snippets":
        count = result.get("total_snippets", 0)
        responses = [
            f"Retrieved {count} useful code patterns.",
            f"Found {count} solid script templates.",
            f"Got {count} proven code examples.",
            f"Collected {count} helpful snippets."
        ]
        return responses[hash_val % len(responses)]
    
    elif tool_name == "create_asset":
        asset_type = result.get("asset_type", "asset")
        name = result.get("name", "")
        responses = [
            f"Created {asset_type} '{name}' successfully.",
            f"New {asset_type} '{name}' is ready.",
            f"Built the {asset_type} '{name}' you needed.",
            f"Asset complete - {asset_type} '{name}' added to project."
        ]
        return responses[hash_val % len(responses)]
    
    elif tool_name == "scene_management":
        message = result.get("message", "Scene operation completed")
        responses = [
            message,
            "Scene setup complete.",
            "Scene configuration done.",
            "Scene work finished."
        ]
        return responses[hash_val % len(responses)]
    
    elif tool_name == "edit_project_config":
        section = result.get("config_section", "settings")
        responses = [
            f"Updated {section} configuration.",
            f"Applied changes to {section} settings.",
            f"Configuration {section} modified.",
            f"Settings for {section} adjusted."
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