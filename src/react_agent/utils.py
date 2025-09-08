"""Enhanced utility & helper functions for the ReAct agent."""

import asyncio
import hashlib
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from datetime import datetime, UTC
import json

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage

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


def extract_tool_calls(message: AIMessage) -> List[Dict[str, Any]]:
    """Extract tool calls from an AI message.
    
    Args:
        message: AIMessage potentially containing tool calls
        
    Returns:
        List of tool call dictionaries with name and arguments
    """
    if not message.tool_calls:
        return []
    
    return [
        {
            "name": call.get("name"),
            "arguments": call.get("args", {}),
            "id": call.get("id")
        }
        for call in message.tool_calls
    ]


def find_last_tool_result(messages: List[BaseMessage]) -> Optional[Dict[str, Any]]:
    """Find the most recent tool execution result in messages.
    
    Args:
        messages: List of conversation messages
        
    Returns:
        Dictionary containing tool result or None
    """
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            try:
                # Try to parse as JSON if possible
                content = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                return {
                    "tool": msg.name,
                    "content": content,
                    "success": not msg.additional_kwargs.get("is_error", False)
                }
            except json.JSONDecodeError:
                return {
                    "tool": msg.name,
                    "content": msg.content,
                    "success": not msg.additional_kwargs.get("is_error", False)
                }
    return None


def summarize_conversation(messages: List[BaseMessage], max_length: int = 500) -> str:
    """Create a concise summary of the conversation history.
    
    Args:
        messages: List of conversation messages
        max_length: Maximum length of summary
        
    Returns:
        String summary of the conversation
    """
    summary_parts = []
    
    for msg in messages[-10:]:  # Last 10 messages
        if isinstance(msg, HumanMessage):
            summary_parts.append(f"User: {get_message_text(msg)[:100]}")
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                tools = [call.get("name") for call in msg.tool_calls]
                summary_parts.append(f"AI: Called tools: {', '.join(tools)}")
            else:
                summary_parts.append(f"AI: {get_message_text(msg)[:100]}")
        elif isinstance(msg, ToolMessage):
            summary_parts.append(f"Tool ({msg.name}): {str(msg.content)[:50]}")
    
    summary = "\n".join(summary_parts)
    if len(summary) > max_length:
        summary = summary[:max_length] + "..."
    
    return summary


def create_step_fingerprint(step_description: str, context: Dict[str, Any]) -> str:
    """Create a unique fingerprint for a step to enable caching.
    
    Args:
        step_description: Description of the step
        context: Additional context that affects the step
        
    Returns:
        Hexadecimal fingerprint string
    """
    # Combine step description with relevant context
    fingerprint_data = {
        "description": step_description,
        "context": {k: v for k, v in context.items() if k in ["tools", "constraints"]}
    }
    
    # Create hash
    json_str = json.dumps(fingerprint_data, sort_keys=True)
    return hashlib.md5(json_str.encode()).hexdigest()


async def parallel_tool_execution(
    tool_calls: List[Dict[str, Any]], 
    tools_dict: Dict[str, Any],
    timeout: int = 30
) -> List[Dict[str, Any]]:
    """Execute multiple tool calls in parallel.
    
    Args:
        tool_calls: List of tool call specifications
        tools_dict: Dictionary mapping tool names to tool functions
        timeout: Timeout for each tool execution
        
    Returns:
        List of results from tool executions
    """
    async def execute_tool(call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single tool call with timeout."""
        tool_name = call.get("name")
        tool_args = call.get("arguments", {})
        
        if tool_name not in tools_dict:
            return {
                "error": f"Tool {tool_name} not found",
                "tool": tool_name
            }
        
        try:
            tool = tools_dict[tool_name]
            result = await asyncio.wait_for(
                tool(**tool_args),
                timeout=timeout
            )
            return {
                "result": result,
                "tool": tool_name,
                "success": True
            }
        except asyncio.TimeoutError:
            return {
                "error": f"Tool {tool_name} timed out",
                "tool": tool_name,
                "success": False
            }
        except Exception as e:
            return {
                "error": f"Tool {tool_name} failed: {str(e)}",
                "tool": tool_name,
                "success": False
            }
    
    # Execute all tools in parallel
    tasks = [execute_tool(call) for call in tool_calls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any exceptions that weren't caught
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            final_results.append({
                "error": f"Unexpected error: {str(result)}",
                "tool": tool_calls[i].get("name"),
                "success": False
            })
        else:
            final_results.append(result)
    
    return final_results


def validate_plan_dependencies(steps: List[Any]) -> List[str]:
    """Validate that plan step dependencies are valid.
    
    Args:
        steps: List of plan steps with potential dependencies
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    for i, step in enumerate(steps):
        if hasattr(step, 'dependencies'):
            for dep in step.dependencies:
                if dep >= i:
                    errors.append(f"Step {i} depends on future step {dep}")
                if dep < 0:
                    errors.append(f"Step {i} has invalid dependency {dep}")
                if dep >= len(steps):
                    errors.append(f"Step {i} depends on non-existent step {dep}")
    
    return errors


def format_tool_response(tool_name: str, response: Any, max_length: int = 1000) -> str:
    """Format a tool response for display or assessment.
    
    Args:
        tool_name: Name of the tool
        response: Raw response from the tool
        max_length: Maximum length of formatted response
        
    Returns:
        Formatted string representation
    """
    if isinstance(response, dict):
        if "error" in response:
            return f"[{tool_name}] Error: {response['error']}"
        
        # Format successful response
        if "result" in response:
            result = response["result"]
            if isinstance(result, dict):
                # Pretty print JSON-like results
                formatted = json.dumps(result, indent=2)
                if len(formatted) > max_length:
                    formatted = formatted[:max_length] + "..."
                return f"[{tool_name}] Success:\n{formatted}"
            else:
                result_str = str(result)
                if len(result_str) > max_length:
                    result_str = result_str[:max_length] + "..."
                return f"[{tool_name}] Success: {result_str}"
    
    # Fallback to string representation
    result_str = str(response)
    if len(result_str) > max_length:
        result_str = result_str[:max_length] + "..."
    return f"[{tool_name}]: {result_str}"


def calculate_step_complexity(step_description: str, tool_name: Optional[str] = None) -> float:
    """Calculate complexity score for a plan step.
    
    Args:
        step_description: Description of the step
        tool_name: Optional tool specified for the step
        
    Returns:
        Complexity score between 0.0 and 1.0
    """
    complexity = 0.3  # Base complexity
    
    # Adjust based on description length and complexity
    if len(step_description) > 200:
        complexity += 0.2
    
    # Check for complex keywords
    complex_keywords = ["multiple", "analyze", "compare", "evaluate", "determine", "assess"]
    for keyword in complex_keywords:
        if keyword.lower() in step_description.lower():
            complexity += 0.1
    
    # Adjust based on tool
    if tool_name:
        tool_complexity = {
            "search": 0.2,
            "calculate": 0.1,
            "get_current_time": 0.0,
            "store_memory": 0.1,
            "retrieve_memory": 0.1,
        }
        complexity += tool_complexity.get(tool_name, 0.15)
    
    return min(complexity, 1.0)


def estimate_step_duration(step_description: str, tool_name: Optional[str] = None) -> int:
    """Estimate duration in seconds for a step execution.
    
    Args:
        step_description: Description of the step
        tool_name: Optional tool specified for the step
        
    Returns:
        Estimated duration in seconds
    """
    # Base duration
    duration = 2
    
    # Tool-specific estimates
    if tool_name:
        tool_durations = {
            "search": 5,
            "calculate": 1,
            "get_current_time": 1,
            "store_memory": 1,
            "retrieve_memory": 1,
            "validate_json": 1,
            "text_analysis": 2,
        }
        duration = tool_durations.get(tool_name, 3)
    
    # Adjust based on complexity
    complexity = calculate_step_complexity(step_description, tool_name)
    duration = int(duration * (1 + complexity))
    
    return duration


class MessageBuffer:
    """Buffer for managing conversation history with size limits."""
    
    def __init__(self, max_messages: int = 50, max_tokens: int = 10000):
        """Initialize message buffer.
        
        Args:
            max_messages: Maximum number of messages to keep
            max_tokens: Approximate maximum token count
        """
        self.messages: List[BaseMessage] = []
        self.max_messages = max_messages
        self.max_tokens = max_tokens
    
    def add(self, message: BaseMessage) -> None:
        """Add a message to the buffer."""
        self.messages.append(message)
        self._trim()
    
    def _trim(self) -> None:
        """Trim buffer to stay within limits."""
        # Trim by message count
        if len(self.messages) > self.max_messages:
            # Keep first message (usually system) and recent messages
            if self.messages:
                self.messages = [self.messages[0]] + self.messages[-(self.max_messages-1):]
        
        # Trim by approximate token count (4 chars ≈ 1 token)
        total_chars = sum(len(get_message_text(msg)) for msg in self.messages)
        
        while total_chars > self.max_tokens * 4 and len(self.messages) > 2:
            # Remove messages from the middle
            mid_index = len(self.messages) // 2
            self.messages.pop(mid_index)
            total_chars = sum(len(get_message_text(msg)) for msg in self.messages)
    
    def get_messages(self, last_n: Optional[int] = None) -> List[BaseMessage]:
        """Get messages from buffer.
        
        Args:
            last_n: If specified, return only the last N messages
            
        Returns:
            List of messages
        """
        if last_n is None:
            return self.messages.copy()
        return self.messages[-last_n:] if self.messages else []
    
    def clear(self) -> None:
        """Clear all messages from buffer."""
        self.messages = []
    
    def get_summary(self) -> str:
        """Get a summary of buffer contents."""
        return f"Buffer: {len(self.messages)} messages, ~{sum(len(get_message_text(msg)) for msg in self.messages)//4} tokens"


class PerformanceTracker:
    """Track performance metrics for the agent."""
    
    def __init__(self):
        """Initialize performance tracker."""
        self.metrics: Dict[str, Any] = {
            "total_steps": 0,
            "successful_steps": 0,
            "failed_steps": 0,
            "total_tool_calls": 0,
            "tool_errors": 0,
            "total_assessments": 0,
            "total_replans": 0,
            "start_time": datetime.now(UTC),
            "step_durations": [],
            "tool_durations": [],
        }
    
    def record_step(self, success: bool, duration: float) -> None:
        """Record a step execution."""
        self.metrics["total_steps"] += 1
        if success:
            self.metrics["successful_steps"] += 1
        else:
            self.metrics["failed_steps"] += 1
        self.metrics["step_durations"].append(duration)
    
    def record_tool_call(self, success: bool, duration: float) -> None:
        """Record a tool call."""
        self.metrics["total_tool_calls"] += 1
        if not success:
            self.metrics["tool_errors"] += 1
        self.metrics["tool_durations"].append(duration)
    
    def record_assessment(self) -> None:
        """Record an assessment."""
        self.metrics["total_assessments"] += 1
    
    def record_replan(self) -> None:
        """Record a replan event."""
        self.metrics["total_replans"] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        elapsed_time = (datetime.now(UTC) - self.metrics["start_time"]).total_seconds()
        
        return {
            "elapsed_time": elapsed_time,
            "total_steps": self.metrics["total_steps"],
            "success_rate": (
                self.metrics["successful_steps"] / self.metrics["total_steps"] 
                if self.metrics["total_steps"] > 0 else 0
            ),
            "avg_step_duration": (
                sum(self.metrics["step_durations"]) / len(self.metrics["step_durations"])
                if self.metrics["step_durations"] else 0
            ),
            "total_tool_calls": self.metrics["total_tool_calls"],
            "tool_error_rate": (
                self.metrics["tool_errors"] / self.metrics["total_tool_calls"]
                if self.metrics["total_tool_calls"] > 0 else 0
            ),
            "replans": self.metrics["total_replans"],
        }


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
        "Done with this part."
    ]
    return generic_responses[hash_val % len(generic_responses)]


def extract_message_content(msg: BaseMessage) -> str:
    """Extract message content - alias for get_message_text for backwards compatibility."""
    return get_message_text(msg)


def format_plan_for_display(plan: Any) -> str:
    """Format an execution plan for display - alias for format_plan_visualization."""
    return format_plan_visualization(plan)


def create_user_friendly_step_message(step: Any, step_num: int, total_steps: int, goal: str) -> str:
    """Create user-friendly step message - alias for create_dynamic_step_message."""
    return create_dynamic_step_message(step, step_num, total_steps, goal)