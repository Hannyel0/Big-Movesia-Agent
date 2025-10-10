"""File approval handler node - handles human-in-the-loop for file operations."""

from __future__ import annotations
from typing import Any, Dict
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import interrupt
from langgraph.runtime import Runtime
from react_agent.context import Context
from react_agent.state import State
from react_agent.tools.file_operation import execute_file_operation


async def check_file_approval(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """
    Check if last tool call needs approval and trigger interrupt if needed.
    
    This node is called when a file operation requires human approval.
    It triggers an interrupt to get human input, then either executes
    the approved operation or adds a rejection message.
    """
    print("📋 [FileApproval] Checking for file operation approval...")
    
    # Find the last tool message
    last_tool_message = None
    for msg in reversed(state.messages):
        if isinstance(msg, ToolMessage):
            last_tool_message = msg
            break
    
    if not last_tool_message:
        # No tool message, continue normally
        print("⚠️  [FileApproval] No tool message found, skipping approval")
        return {}
    
    # Parse tool result
    try:
        import json
        result = json.loads(last_tool_message.content)
    except Exception as e:
        print(f"⚠️  [FileApproval] Could not parse tool result: {e}")
        return {}
    
    # Check if approval is needed
    if not result.get("needs_approval"):
        print("ℹ️  [FileApproval] Tool result does not need approval, skipping")
        return {}
    
    # Get approval data
    approval_data = result.get("approval_data", {})
    pending_operation = result.get("pending_operation", {})
    
    print(f"🔔 [FileApproval] Triggering interrupt for approval:")
    print(f"   Operation: {approval_data.get('operation')}")
    print(f"   File: {approval_data.get('file_path')}")
    
    # Trigger interrupt for human approval
    approval_result = interrupt(approval_data)
    
    # Handle different response formats
    # LangGraph can return: {"approved": True}, True, or None
    approved = False
    if isinstance(approval_result, dict):
        approved = approval_result.get("approved", False)
    elif isinstance(approval_result, bool):
        approved = approval_result
    
    print(f"📝 [FileApproval] Approval result: {approved}")
    
    # Check if approved
    if not approved:
        # User rejected - add rejection message
        print(f"❌ [FileApproval] Operation rejected by user")
        rejection_msg = AIMessage(
            content=f"❌ File operation cancelled: {approval_data.get('message', 'User rejected the operation')}"
        )
        return {
            "messages": [rejection_msg]
        }
    
    # User approved - execute the operation
    print(f"✅ [FileApproval] Operation approved, executing...")
    execution_result = await execute_file_operation(pending_operation)
    
    # Create new tool message with execution result
    tool_msg = ToolMessage(
        content=json.dumps(execution_result),
        tool_call_id=last_tool_message.tool_call_id,
        name=last_tool_message.name
    )
    
    # Create AI message confirming the operation
    if execution_result.get("success"):
        file_path = execution_result.get("file_path", 
                                       execution_result.get("to_path", "file"))
        operation = execution_result.get("operation", "operation")
        
        confirmation_msgs = {
            "write": f"✅ File {'created' if execution_result.get('created') else 'updated'}: {file_path}",
            "modify": f"✅ File modified: {file_path}",
            "delete": f"✅ File deleted: {file_path}",
            "move": f"✅ File moved: {execution_result.get('from_path')} → {file_path}"
        }
        
        confirmation_content = confirmation_msgs.get(operation, f"✅ File operation completed: {file_path}")
        print(f"✅ [FileApproval] {confirmation_content}")
        
        ai_msg = AIMessage(content=confirmation_content)
    else:
        error_msg = f"❌ File operation failed: {execution_result.get('error', 'Unknown error')}"
        print(f"❌ [FileApproval] {error_msg}")
        ai_msg = AIMessage(content=error_msg)
    
    return {
        "messages": [tool_msg, ai_msg]
    }
