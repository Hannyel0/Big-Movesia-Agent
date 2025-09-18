# Intelligent Error Recovery System Integration Guide

## Overview

This implementation adds sophisticated error recovery capabilities to your ReAct agent. When a tool call fails, the system now:

1. **Intelligently diagnoses** the error using LLM analysis
2. **Creates targeted recovery plans** to fix the root cause  
3. **Executes recovery steps** systematically
4. **Returns to the original failed step** after fixing the issue

## Key Components

### 1. Error Recovery Node (`error_recovery.py`)
- **`ErrorDiagnosis`**: Categorizes errors (missing dependencies, configuration issues, etc.)
- **`ErrorRecoveryPlan`**: Creates systematic fix plans
- **`execute_error_recovery()`**: Main recovery orchestrator

### 2. Enhanced Assessment Node
- Detects critical errors that need systematic resolution
- Triggers error recovery instead of simple retries
- Uses `_is_critical_error()` to identify recoverable issues

### 3. Enhanced Routing System
- New route: `route_after_assess()` → `"error_recovery"`
- Recovery-aware routing that handles inserted recovery steps
- Tracks when agent is in recovery mode vs normal execution

### 4. Updated Graph Builder
- Adds `error_recovery` node to the graph
- Connects error recovery with proper routing

## How It Works

### Normal Flow
```
act → tools → assess → advance_step → act (next step)
```

### Error Recovery Flow
```
act → tools → assess → error_recovery → act (recovery steps) → assess → advance_step → act (retry original)
```

### Detailed Process

1. **Error Detection**: Assessment node detects critical error
2. **Diagnosis**: LLM analyzes the error and categorizes it
3. **Recovery Planning**: Creates 1-3 targeted steps to fix the issue
4. **Plan Insertion**: Inserts recovery steps before the failed step
5. **Recovery Execution**: Executes recovery steps systematically
6. **Original Retry**: Returns to original failed step after fixes

## Error Categories Handled

- **Dependency Missing**: Creates missing packages/references
- **Configuration Error**: Updates project settings
- **Resource Not Found**: Creates missing assets/files
- **Build Error**: Fixes compilation issues
- **Project State Error**: Resolves Unity/Unreal project issues

## File Updates Required

### 1. Create New File: `react_agent/graph/nodes/error_recovery.py`
Copy the error recovery system code.

### 2. Update: `react_agent/graph/nodes/assess.py`
Replace with enhanced assessment node that detects critical errors.

### 3. Update: `react_agent/graph/routing.py`
Replace with enhanced routing that handles error recovery.

### 4. Update: `react_agent/graph/builder.py`
Replace with enhanced graph builder that includes error recovery node.

### 5. Update State Schema (Optional Enhancement)
Add to `react_agent/state.py`:
```python
@dataclass
class State(InputState):
    # ... existing fields ...
    
    # Error recovery tracking
    error_recovery_active: bool = field(default=False)
    needs_error_recovery: bool = field(default=False)
    error_context: Optional[Dict[str, Any]] = field(default=None)
```

## Configuration Options

Add to `react_agent/context.py`:
```python
@dataclass(kw_only=True) 
class Context:
    # ... existing fields ...
    
    # Error recovery settings
    enable_error_recovery: bool = field(
        default=True,
        metadata={"description": "Enable intelligent error recovery system"}
    )
    
    max_recovery_attempts: int = field(
        default=2,
        metadata={"description": "Maximum error recovery attempts per issue"}
    )
    
    error_recovery_timeout: int = field(
        default=60,
        metadata={"description": "Timeout for error recovery operations"}
    )
```

## Benefits

1. **Intelligent Problem-Solving**: Instead of blind retries, analyzes and fixes root causes
2. **Systematic Recovery**: Creates targeted plans to address specific error types
3. **Transparent Process**: Clearly communicates what went wrong and how it's being fixed
4. **Reduced Failures**: Addresses underlying issues before they cause repeated failures
5. **Better User Experience**: Proactive error resolution with clear explanations

## Example Scenarios

### Scenario 1: Missing Dependency
```
Original: write_file fails (missing Unity package)
Recovery: search → create_asset (install package) → retry write_file
```

### Scenario 2: Configuration Error  
```
Original: compile_and_test fails (build settings)
Recovery: get_project_info → edit_project_config → retry compile_and_test
```

### Scenario 3: Resource Not Found
```
Original: scene_management fails (missing scene)
Recovery: create_asset (scene) → retry scene_management
```

## Testing the Implementation

1. **Trigger intentional errors** in your tools to test recovery
2. **Monitor recovery plan generation** - ensure plans address root causes
3. **Verify original step retry** - check that failed steps are retried after recovery
4. **Test different error categories** - ensure appropriate recovery for each type

The system provides robust, intelligent error handling that significantly improves agent reliability and user experience.