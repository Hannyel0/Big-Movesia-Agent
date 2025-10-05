"""File operation tool with validation and diff generation."""

from __future__ import annotations
from typing import Literal, Optional, Dict, Any
import difflib
from pathlib import Path

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import get_runtime

from react_agent.context import Context


@tool
async def file_operation(
    operation: Literal["read", "write", "modify", "delete", "move", "diff"],
    file_path: str,
    config: RunnableConfig,
    content: Optional[str] = None,
    modification_spec: Optional[Dict[str, Any]] = None,
    validate_only: bool = False
) -> Dict[str, Any]:
    """Unified file I/O with Unity AssetDatabase integration.
    
    Handles all file manipulation safely with validation, diff generation,
    and Unity-compatible operations.
    
    Args:
        operation: Type of operation (read/write/modify/delete/move/diff)
        file_path: Target file path (relative to project root)
        content: Content for write/modify operations
        modification_spec: Specification for surgical edits (line_ranges, patterns, replacements)
        validate_only: If True, performs dry-run without actual changes
        
    Returns:
        Operation result with diff preview and validation status
    """
    try:
        runtime = get_runtime(Context)
        
        # Get project context
        configurable = config.get("configurable", {})
        project_root = configurable.get("project_root")
        
        if not project_root:
            return {
                "success": False,
                "error": "Project root not available. Ensure Unity project is connected."
            }
        
        # Resolve absolute path
        abs_path = Path(project_root) / file_path
        
        # Validate path is within project
        try:
            abs_path = abs_path.resolve()
            project_root_resolved = Path(project_root).resolve()
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
        
        # Execute operation
        if operation == "read":
            return await _file_read(abs_path, file_path)
        
        elif operation == "write":
            if not content:
                return {"success": False, "error": "Content required for write operation"}
            return await _file_write(abs_path, file_path, content, validate_only)
        
        elif operation == "modify":
            if not modification_spec:
                return {"success": False, "error": "Modification spec required for modify operation"}
            return await _file_modify(abs_path, file_path, modification_spec, validate_only)
        
        elif operation == "delete":
            return await _file_delete(abs_path, file_path, validate_only)
        
        elif operation == "move":
            if not modification_spec or "new_path" not in modification_spec:
                return {"success": False, "error": "new_path required in modification_spec for move"}
            new_path = Path(project_root) / modification_spec["new_path"]
            return await _file_move(abs_path, new_path, file_path, modification_spec["new_path"], validate_only)
        
        elif operation == "diff":
            if not content:
                return {"success": False, "error": "Content required for diff operation"}
            return await _file_diff(abs_path, file_path, content)
        
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}
        
    except Exception as e:
        return {
            "success": False,
            "error": f"File operation failed: {str(e)}",
            "operation": operation,
            "file_path": file_path
        }


async def _file_read(abs_path: Path, rel_path: str) -> Dict[str, Any]:
    """Read file contents."""
    if not abs_path.exists():
        return {
            "success": False,
            "error": f"File not found: {rel_path}"
        }
    
    try:
        content = abs_path.read_text(encoding="utf-8")
        return {
            "success": True,
            "operation": "read",
            "file_path": rel_path,
            "content": content,
            "size_bytes": len(content.encode("utf-8")),
            "line_count": len(content.split("\n"))
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to read file: {str(e)}",
            "file_path": rel_path
        }


async def _file_write(abs_path: Path, rel_path: str, content: str, validate_only: bool) -> Dict[str, Any]:
    """Write content to file."""
    # Generate diff if file exists
    diff = ""
    if abs_path.exists():
        old_content = abs_path.read_text(encoding="utf-8")
        diff = "\n".join(difflib.unified_diff(
            old_content.splitlines(),
            content.splitlines(),
            fromfile=f"{rel_path} (original)",
            tofile=f"{rel_path} (new)",
            lineterm=""
        ))
    
    if validate_only:
        return {
            "success": True,
            "operation": "write",
            "file_path": rel_path,
            "validate_only": True,
            "would_create": not abs_path.exists(),
            "diff_preview": diff,
            "size_bytes": len(content.encode("utf-8"))
        }
    
    try:
        # Ensure parent directory exists
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        abs_path.write_text(content, encoding="utf-8")
        
        return {
            "success": True,
            "operation": "write",
            "file_path": rel_path,
            "created": not diff,
            "diff": diff,
            "size_bytes": len(content.encode("utf-8")),
            "line_count": len(content.split("\n"))
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to write file: {str(e)}",
            "file_path": rel_path
        }


async def _file_modify(abs_path: Path, rel_path: str, spec: Dict[str, Any], validate_only: bool) -> Dict[str, Any]:
    """Modify file with surgical edits."""
    if not abs_path.exists():
        return {
            "success": False,
            "error": f"File not found: {rel_path}"
        }
    
    try:
        content = abs_path.read_text(encoding="utf-8")
        lines = content.split("\n")
        modified_lines = lines.copy()
        
        # Apply modifications based on spec
        if "line_ranges" in spec:
            # Modify specific line ranges
            for line_range in spec["line_ranges"]:
                start = line_range.get("start", 1) - 1  # Convert to 0-indexed
                end = line_range.get("end", len(lines)) - 1
                replacement = line_range.get("replacement", "")
                
                if 0 <= start <= end < len(modified_lines):
                    modified_lines[start:end+1] = [replacement]
        
        elif "pattern_replacements" in spec:
            # Pattern-based replacements
            import re
            for pattern_spec in spec["pattern_replacements"]:
                pattern = pattern_spec.get("pattern")
                replacement = pattern_spec.get("replacement", "")
                
                if pattern:
                    modified_content = "\n".join(modified_lines)
                    modified_content = re.sub(pattern, replacement, modified_content)
                    modified_lines = modified_content.split("\n")
        
        new_content = "\n".join(modified_lines)
        
        # Generate diff
        diff = "\n".join(difflib.unified_diff(
            lines,
            modified_lines,
            fromfile=f"{rel_path} (original)",
            tofile=f"{rel_path} (modified)",
            lineterm=""
        ))
        
        if validate_only:
            return {
                "success": True,
                "operation": "modify",
                "file_path": rel_path,
                "validate_only": True,
                "diff_preview": diff
            }
        
        # Apply changes
        abs_path.write_text(new_content, encoding="utf-8")
        
        return {
            "success": True,
            "operation": "modify",
            "file_path": rel_path,
            "diff": diff,
            "modifications_applied": len(spec.get("line_ranges", [])) + len(spec.get("pattern_replacements", []))
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to modify file: {str(e)}",
            "file_path": rel_path
        }


async def _file_delete(abs_path: Path, rel_path: str, validate_only: bool) -> Dict[str, Any]:
    """Delete file."""
    if not abs_path.exists():
        return {
            "success": False,
            "error": f"File not found: {rel_path}"
        }
    
    if validate_only:
        return {
            "success": True,
            "operation": "delete",
            "file_path": rel_path,
            "validate_only": True,
            "would_delete": True
        }
    
    try:
        abs_path.unlink()
        return {
            "success": True,
            "operation": "delete",
            "file_path": rel_path,
            "deleted": True
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to delete file: {str(e)}",
            "file_path": rel_path
        }


async def _file_move(abs_path: Path, new_abs_path: Path, rel_path: str, new_rel_path: str, validate_only: bool) -> Dict[str, Any]:
    """Move/rename file."""
    if not abs_path.exists():
        return {
            "success": False,
            "error": f"Source file not found: {rel_path}"
        }
    
    if new_abs_path.exists():
        return {
            "success": False,
            "error": f"Destination already exists: {new_rel_path}"
        }
    
    if validate_only:
        return {
            "success": True,
            "operation": "move",
            "from_path": rel_path,
            "to_path": new_rel_path,
            "validate_only": True,
            "would_move": True
        }
    
    try:
        # Ensure destination directory exists
        new_abs_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Move file
        abs_path.rename(new_abs_path)
        
        return {
            "success": True,
            "operation": "move",
            "from_path": rel_path,
            "to_path": new_rel_path,
            "moved": True
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to move file: {str(e)}",
            "from_path": rel_path,
            "to_path": new_rel_path
        }


async def _file_diff(abs_path: Path, rel_path: str, new_content: str) -> Dict[str, Any]:
    """Generate diff without modifying file."""
    if not abs_path.exists():
        return {
            "success": True,
            "operation": "diff",
            "file_path": rel_path,
            "file_exists": False,
            "diff": f"New file would be created:\n{new_content}"
        }
    
    try:
        old_content = abs_path.read_text(encoding="utf-8")
        diff = "\n".join(difflib.unified_diff(
            old_content.splitlines(),
            new_content.splitlines(),
            fromfile=f"{rel_path} (current)",
            tofile=f"{rel_path} (proposed)",
            lineterm=""
        ))
        
        return {
            "success": True,
            "operation": "diff",
            "file_path": rel_path,
            "diff": diff,
            "changes_detected": bool(diff)
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to generate diff: {str(e)}",
            "file_path": rel_path
        }
