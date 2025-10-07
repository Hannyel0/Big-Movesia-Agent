"""File operation tool - generates approval requests, doesn't handle interrupts."""

from __future__ import annotations
from typing import Literal, Optional, Dict, Any
import difflib
import asyncio
from pathlib import Path
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig


@tool
async def file_operation(
    operation: Literal["read", "write", "modify", "delete", "move", "diff"],
    file_path: str,
    config: RunnableConfig,
    content: Optional[str] = None,
    modification_spec: Optional[Dict[str, Any]] = None,
    new_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Unified file I/O with Unity AssetDatabase integration.
    
    For destructive operations, this tool returns a special response that
    triggers human approval in the graph routing logic.
    
    Args:
        operation: Type of operation (read/write/modify/delete/move/diff)
        file_path: Target file path (relative to project root)
        content: Content for write/modify operations
        modification_spec: Specification for surgical edits
        new_path: Destination path for move operations
        
    Returns:
        Operation result, or approval request for destructive operations
    """
    configurable = config.get("configurable", {})
    project_root = configurable.get("project_root")
    
    if not project_root:
        return {
            "success": False,
            "error": "Project root not available. Ensure Unity project is connected."
        }
    
    abs_path = Path(project_root) / file_path
    
    # Validate path is within project
    try:
        abs_path = await asyncio.to_thread(lambda: abs_path.resolve())
        project_root_resolved = await asyncio.to_thread(lambda: Path(project_root).resolve())
        
        if not str(abs_path).startswith(str(project_root_resolved)):
            return {
                "success": False,
                "error": f"Path {file_path} is outside project root"
            }
    except Exception as e:
        return {
            "success": False,
            "error": f"Invalid file path: {str(e)}"
        }
    
    # Route to appropriate handler
    if operation == "read":
        return await _file_read(abs_path, file_path)
    
    elif operation == "diff":
        if not content:
            return {"success": False, "error": "Content required for diff operation"}
        return await _file_diff(abs_path, file_path, content)
    
    elif operation == "write":
        if not content:
            return {"success": False, "error": "Content required for write operation"}
        return await _file_write_prepare(abs_path, file_path, content)
    
    elif operation == "modify":
        if not modification_spec:
            return {"success": False, "error": "Modification spec required"}
        return await _file_modify_prepare(abs_path, file_path, modification_spec)
    
    elif operation == "delete":
        return await _file_delete_prepare(abs_path, file_path)
    
    elif operation == "move":
        if not new_path:
            return {"success": False, "error": "new_path required for move"}
        new_abs_path = Path(project_root) / new_path
        return await _file_move_prepare(abs_path, new_abs_path, file_path, new_path)
    
    else:
        return {"success": False, "error": f"Unknown operation: {operation}"}


async def _file_read(abs_path: Path, rel_path: str) -> Dict[str, Any]:
    """Read file contents - no approval needed."""
    exists = await asyncio.to_thread(abs_path.exists)
    if not exists:
        return {"success": False, "error": f"File not found: {rel_path}"}
    
    try:
        content = await asyncio.to_thread(abs_path.read_text, encoding="utf-8")
        return {
            "success": True,
            "operation": "read",
            "file_path": rel_path,
            "content": content,
            "size_bytes": len(content.encode("utf-8")),
            "line_count": len(content.split("\n"))
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to read file: {str(e)}"}


async def _file_write_prepare(abs_path: Path, rel_path: str, content: str) -> Dict[str, Any]:
    """Prepare write operation - return approval request."""
    file_exists = await asyncio.to_thread(abs_path.exists)
    
    old_content = ""
    if file_exists:
        old_content = await asyncio.to_thread(abs_path.read_text, encoding="utf-8")
    
    unified_diff = "\n".join(difflib.unified_diff(
        old_content.splitlines(keepends=True),
        content.splitlines(keepends=True),
        fromfile=f"{rel_path} (current)" if file_exists else "/dev/null",
        tofile=f"{rel_path} (proposed)",
        lineterm=""
    ))
    
    # Return approval request instead of calling interrupt
    return {
        "success": True,
        "needs_approval": True,
        "approval_data": {
            "type": "file_operation_approval",
            "operation": "write",
            "file_path": rel_path,
            "file_exists": file_exists,
            "diff": unified_diff,
            "old_content": old_content if file_exists else None,
            "new_content": content,
            "language": _detect_language(rel_path),
            "message": f"{'Update' if file_exists else 'Create'} {rel_path}?"
        },
        # Store data needed to complete the operation
        "pending_operation": {
            "operation": "write",
            "abs_path": str(abs_path),
            "rel_path": rel_path,
            "content": content,
            "file_exists": file_exists
        }
    }


async def _file_modify_prepare(abs_path: Path, rel_path: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare modify operation - return approval request."""
    exists = await asyncio.to_thread(abs_path.exists)
    if not exists:
        return {"success": False, "error": f"File not found: {rel_path}"}
    
    content = await asyncio.to_thread(abs_path.read_text, encoding="utf-8")
    lines = content.split("\n")
    modified_lines = lines.copy()
    
    # Apply modifications to generate preview
    if "line_ranges" in spec:
        for line_range in spec["line_ranges"]:
            start = line_range.get("start", 1) - 1
            end = line_range.get("end", len(lines)) - 1
            replacement = line_range.get("replacement", "")
            
            if 0 <= start <= end < len(modified_lines):
                modified_lines[start:end+1] = [replacement]
    
    elif "pattern_replacements" in spec:
        import re
        for pattern_spec in spec["pattern_replacements"]:
            pattern = pattern_spec.get("pattern")
            replacement = pattern_spec.get("replacement", "")
            if pattern:
                modified_content = "\n".join(modified_lines)
                modified_content = re.sub(pattern, replacement, modified_content)
                modified_lines = modified_content.split("\n")
    
    new_content = "\n".join(modified_lines)
    
    unified_diff = "\n".join(difflib.unified_diff(
        lines,
        modified_lines,
        fromfile=f"{rel_path} (current)",
        tofile=f"{rel_path} (modified)",
        lineterm=""
    ))
    
    return {
        "success": True,
        "needs_approval": True,
        "approval_data": {
            "type": "file_operation_approval",
            "operation": "modify",
            "file_path": rel_path,
            "diff": unified_diff,
            "old_content": content,
            "new_content": new_content,
            "language": _detect_language(rel_path),
            "message": f"Apply modifications to {rel_path}?"
        },
        "pending_operation": {
            "operation": "modify",
            "abs_path": str(abs_path),
            "rel_path": rel_path,
            "content": new_content
        }
    }


async def _file_delete_prepare(abs_path: Path, rel_path: str) -> Dict[str, Any]:
    """Prepare delete operation - return approval request."""
    exists = await asyncio.to_thread(abs_path.exists)
    if not exists:
        return {"success": False, "error": f"File not found: {rel_path}"}
    
    try:
        content = await asyncio.to_thread(abs_path.read_text, encoding="utf-8")
    except:
        content = "[Binary file]"
    
    return {
        "success": True,
        "needs_approval": True,
        "approval_data": {
            "type": "file_operation_approval",
            "operation": "delete",
            "file_path": rel_path,
            "content_preview": content[:500] if len(content) > 500 else content,
            "language": _detect_language(rel_path),
            "message": f"Delete {rel_path}?"
        },
        "pending_operation": {
            "operation": "delete",
            "abs_path": str(abs_path),
            "rel_path": rel_path
        }
    }


async def _file_move_prepare(abs_path: Path, new_abs_path: Path, rel_path: str, new_rel_path: str) -> Dict[str, Any]:
    """Prepare move operation - return approval request."""
    exists = await asyncio.to_thread(abs_path.exists)
    if not exists:
        return {"success": False, "error": f"Source file not found: {rel_path}"}
    
    dest_exists = await asyncio.to_thread(new_abs_path.exists)
    if dest_exists:
        return {"success": False, "error": f"Destination already exists: {new_rel_path}"}
    
    return {
        "success": True,
        "needs_approval": True,
        "approval_data": {
            "type": "file_operation_approval",
            "operation": "move",
            "from_path": rel_path,
            "to_path": new_rel_path,
            "message": f"Move {rel_path} → {new_rel_path}?"
        },
        "pending_operation": {
            "operation": "move",
            "abs_path": str(abs_path),
            "new_abs_path": str(new_abs_path),
            "from_path": rel_path,
            "to_path": new_rel_path
        }
    }


async def _file_diff(abs_path: Path, rel_path: str, new_content: str) -> Dict[str, Any]:
    """Generate diff without modifying file - no approval needed."""
    # Check existence in thread pool
    exists = await asyncio.to_thread(abs_path.exists)
    if not exists:
        return {
            "success": True,
            "operation": "diff",
            "file_path": rel_path,
            "file_exists": False,
            "diff": f"New file would be created:\n{new_content}"
        }
    
    try:
        # Read file in thread pool
        old_content = await asyncio.to_thread(abs_path.read_text, encoding="utf-8")
        unified_diff = "\n".join(difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"{rel_path} (current)",
            tofile=f"{rel_path} (proposed)",
            lineterm=""
        ))
        
        return {
            "success": True,
            "operation": "diff",
            "file_path": rel_path,
            "diff": unified_diff,
            "changes_detected": bool(unified_diff)
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to generate diff: {str(e)}"}


def _detect_language(file_path: str) -> str:
    """Detect programming language from file extension."""
    ext_map = {
        ".cs": "csharp",
        ".js": "javascript",
        ".ts": "typescript",
        ".py": "python",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
        ".java": "java",
        ".json": "json",
        ".xml": "xml",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".md": "markdown",
        ".txt": "plaintext",
        ".shader": "hlsl",
        ".hlsl": "hlsl",
        ".glsl": "glsl",
    }
    
    ext = Path(file_path).suffix.lower()
    return ext_map.get(ext, "plaintext")


# Execution functions (called after approval)
async def execute_file_operation(pending_op: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a file operation after approval."""
    operation = pending_op["operation"]
    
    if operation == "write":
        return await _execute_write(pending_op)
    elif operation == "modify":
        return await _execute_modify(pending_op)
    elif operation == "delete":
        return await _execute_delete(pending_op)
    elif operation == "move":
        return await _execute_move(pending_op)
    
    return {"success": False, "error": f"Unknown operation: {operation}"}


async def _execute_write(pending_op: Dict[str, Any]) -> Dict[str, Any]:
    """Execute write operation."""
    abs_path = Path(pending_op["abs_path"])
    rel_path = pending_op["rel_path"]
    content = pending_op["content"]
    file_exists = pending_op["file_exists"]
    
    try:
        await asyncio.to_thread(abs_path.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(abs_path.write_text, content, encoding="utf-8")
        
        return {
            "success": True,
            "operation": "write",
            "file_path": rel_path,
            "created": not file_exists,
            "size_bytes": len(content.encode("utf-8")),
            "line_count": len(content.split("\n"))
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to write file: {str(e)}"}


async def _execute_modify(pending_op: Dict[str, Any]) -> Dict[str, Any]:
    """Execute modify operation."""
    abs_path = Path(pending_op["abs_path"])
    rel_path = pending_op["rel_path"]
    content = pending_op["content"]
    
    try:
        await asyncio.to_thread(abs_path.write_text, content, encoding="utf-8")
        
        return {
            "success": True,
            "operation": "modify",
            "file_path": rel_path
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to modify file: {str(e)}"}


async def _execute_delete(pending_op: Dict[str, Any]) -> Dict[str, Any]:
    """Execute delete operation."""
    abs_path = Path(pending_op["abs_path"])
    rel_path = pending_op["rel_path"]
    
    try:
        await asyncio.to_thread(abs_path.unlink)
        return {
            "success": True,
            "operation": "delete",
            "file_path": rel_path,
            "deleted": True
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to delete file: {str(e)}"}


async def _execute_move(pending_op: Dict[str, Any]) -> Dict[str, Any]:
    """Execute move operation."""
    abs_path = Path(pending_op["abs_path"])
    new_abs_path = Path(pending_op["new_abs_path"])
    from_path = pending_op["from_path"]
    to_path = pending_op["to_path"]
    
    try:
        await asyncio.to_thread(new_abs_path.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(abs_path.rename, new_abs_path)
        
        return {
            "success": True,
            "operation": "move",
            "from_path": from_path,
            "to_path": to_path,
            "moved": True
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to move file: {str(e)}"}