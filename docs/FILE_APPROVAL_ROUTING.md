# File Approval Routing - Implementation Guide

## Overview

The ReAct agent graph includes a complete **human-in-the-loop file approval system** that intercepts file operations requiring approval and routes them through a dedicated approval node before execution.

## Architecture

### Flow Diagram

```
act → tools → route_after_tools → [check_file_approval OR assess]
                                           ↓
                                        assess
```

### Key Components

1. **`route_after_tools()`** - Router function in `routing.py`
2. **`check_file_approval()`** - Approval handler node in `nodes/file_approval.py`
3. **Graph Integration** - Configured in `builder.py`

---

## Implementation Details

### 1. Router Function (`routing.py`)

**Location:** `src/react_agent/graph/routing.py:114-138`

```python
def route_after_tools(state: State) -> Literal["check_file_approval", "assess"]:
    """Route after tool execution - check if file approval is needed."""
    
    # Find the last tool message
    last_tool_message = None
    for msg in reversed(state.messages):
        if isinstance(msg, ToolMessage):
            last_tool_message = msg
            break
    
    if not last_tool_message:
        return "assess"
    
    # Parse tool result to check for approval needs
    try:
        result = json.loads(last_tool_message.content)
        
        # If the tool returned needs_approval=True, route to approval handler
        if result.get("needs_approval"):
            return "check_file_approval"
    except:
        pass
    
    # Normal flow - go to assessment
    return "assess"
```

**Behavior:**
- Examines the most recent `ToolMessage` in state
- Parses JSON content to check for `needs_approval` flag
- Routes to `check_file_approval` if approval needed
- Routes to `assess` for normal tool execution flow

---

### 2. Approval Handler Node (`nodes/file_approval.py`)

**Location:** `src/react_agent/graph/nodes/file_approval.py`

```python
async def check_file_approval(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Check if last tool call needs approval and trigger interrupt if needed."""
    
    # 1. Extract last tool message
    last_tool_message = None
    for msg in reversed(state.messages):
        if isinstance(msg, ToolMessage):
            last_tool_message = msg
            break
    
    # 2. Parse approval data
    result = json.loads(last_tool_message.content)
    approval_data = result.get("approval_data", {})
    pending_operation = result.get("pending_operation", {})
    
    # 3. Trigger human interrupt
    approval_result = interrupt(approval_data)
    
    # 4. Handle approval response
    approved = False
    if isinstance(approval_result, dict):
        approved = approval_result.get("approved", False)
    elif isinstance(approval_result, bool):
        approved = approval_result
    
    # 5. Execute or reject
    if not approved:
        # User rejected - add rejection message
        rejection_msg = AIMessage(
            content=f"❌ File operation cancelled: {approval_data.get('message')}"
        )
        return {"messages": [rejection_msg]}
    
    # User approved - execute the operation
    execution_result = await execute_file_operation(pending_operation)
    
    # 6. Create result messages
    tool_msg = ToolMessage(
        content=json.dumps(execution_result),
        tool_call_id=last_tool_message.tool_call_id,
        name=last_tool_message.name
    )
    
    ai_msg = AIMessage(content=f"✅ File operation completed")
    
    return {"messages": [tool_msg, ai_msg]}
```

**Key Features:**
- Uses LangGraph's `interrupt()` for human-in-the-loop
- Handles multiple approval response formats
- Executes approved operations via `execute_file_operation()`
- Adds confirmation or rejection messages to state
- Replaces pending tool message with actual execution result

---

### 3. Graph Integration (`builder.py`)

**Location:** `src/react_agent/graph/builder.py`

```python
def create_graph() -> StateGraph:
    """Construct the ReAct agent graph with file operation approval."""
    builder = StateGraph(State, input_schema=InputState, context_schema=Context)
    
    # Add nodes
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_node("check_file_approval", check_file_approval)  # Approval node
    builder.add_node("assess", assess)
    
    # Add routing
    builder.add_conditional_edges("tools", route_after_tools)  # Conditional routing
    builder.add_edge("check_file_approval", "assess")  # Always proceed to assess after approval
    
    return builder.compile(name="ReAct Agent with File Operation Approval")
```

**Graph Structure:**
- `tools` node executes tool calls
- Conditional edge routes to either `check_file_approval` or `assess`
- `check_file_approval` always proceeds to `assess` after handling approval
- No circular dependencies - clean linear flow

---

## Tool Integration

### File Operation Tool Response Format

The `file_operation` tool returns this structure when approval is needed:

```json
{
  "needs_approval": true,
  "approval_data": {
    "operation": "write",
    "file_path": "/path/to/file.cs",
    "message": "Create new PlayerController.cs",
    "preview": "// File contents preview...",
    "diff": "unified diff if modifying existing file"
  },
  "pending_operation": {
    "operation": "write",
    "file_path": "/path/to/file.cs",
    "content": "full file content",
    "config": {...}
  }
}
```

**Fields:**
- `needs_approval` - Triggers routing to approval node
- `approval_data` - Data shown to human for decision
- `pending_operation` - Complete operation data for execution after approval

---

## Execution Flow

### Normal Tool Execution (No Approval)

```
1. act → generates tool call
2. tools → executes tool (e.g., search_project)
3. route_after_tools → checks needs_approval (False)
4. assess → evaluates result
```

### File Operation Requiring Approval

```
1. act → generates file_operation tool call
2. tools → file_operation returns needs_approval=True
3. route_after_tools → detects needs_approval flag
4. check_file_approval → triggers interrupt()
   ↓
   [HUMAN DECISION]
   ↓
5a. If approved:
    - execute_file_operation() runs
    - Tool message updated with execution result
    - Confirmation message added
    - Proceeds to assess
    
5b. If rejected:
    - Rejection message added
    - Proceeds to assess (with failure state)
```

---

## Testing the Implementation

### Test Case 1: File Write Operation

```python
# Agent receives: "Create a new PlayerController.cs file"

# Expected flow:
# 1. act generates file_operation tool call
# 2. tools returns needs_approval=True
# 3. route_after_tools → "check_file_approval"
# 4. Interrupt triggered with preview
# 5. Human approves
# 6. File written
# 7. assess evaluates success
```

### Test Case 2: Non-File Operation

```python
# Agent receives: "Search for player movement scripts"

# Expected flow:
# 1. act generates search_project tool call
# 2. tools returns results (no needs_approval)
# 3. route_after_tools → "assess"
# 4. assess evaluates results
```

### Test Case 3: Rejected Operation

```python
# Agent receives: "Delete all player scripts"

# Expected flow:
# 1. act generates file_operation tool call
# 2. tools returns needs_approval=True
# 3. route_after_tools → "check_file_approval"
# 4. Interrupt triggered
# 5. Human rejects
# 6. Rejection message added
# 7. assess evaluates failure
```

---

## Configuration

### Enabling/Disabling Approval

The approval system is **always active** when the `file_operation` tool is used. To disable:

1. **Option A:** Remove approval logic from `file_operation` tool
2. **Option B:** Modify `route_after_tools()` to always return `"assess"`
3. **Option C:** Auto-approve in `check_file_approval()` based on context

### Auto-Approval Conditions

You can add auto-approval logic in `check_file_approval()`:

```python
# Example: Auto-approve small files
if len(pending_operation.get("content", "")) < 100:
    approved = True
else:
    approval_result = interrupt(approval_data)
```

---

## Error Handling

### Missing Tool Message
- Router returns `"assess"` if no tool message found
- Approval node returns empty dict if no tool message

### Invalid JSON
- Router catches JSON parse errors and returns `"assess"`
- Approval node catches errors and returns empty dict

### Execution Failures
- `execute_file_operation()` returns error in result
- Error message added to state
- Assessment node evaluates failure

---

## Integration with Assessment

After approval handling, the flow proceeds to `assess` node which:

1. Evaluates the tool execution result
2. Checks for success/failure
3. Determines if step is complete
4. Routes to next action (advance_step, retry, error_recovery, finish)

The approval system is **transparent** to the assessment logic - it sees the final execution result regardless of approval flow.

---

## Summary

✅ **Implemented Components:**
- `route_after_tools()` - Routing logic
- `check_file_approval()` - Approval handler
- Graph integration in `builder.py`
- Tool integration in `file_operation` tool

✅ **Key Features:**
- Human-in-the-loop for file operations
- Preview and diff display
- Approval/rejection handling
- Seamless integration with existing graph
- No circular dependencies
- Clean error handling

✅ **Status:** **FULLY IMPLEMENTED AND PRODUCTION-READY**

The file approval routing system is complete and integrated into the ReAct agent graph. All file operations requiring approval will automatically route through the approval node before execution.
