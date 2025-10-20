"""File operation tool - generates approval requests, doesn't handle interrupts."""

from __future__ import annotations
from typing import Literal, Optional, Dict, Any
import difflib
import asyncio
import logging
from pathlib import Path
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


# ============================================================================
# SMART PATH RESOLUTION - Makes file_operation intelligent and self-contained
# ============================================================================

async def _resolve_file_path(
    partial_path: str,
    project_root: Path,
    sqlite_path: Optional[Path] = None,
    operation: str = "read"
) -> Dict[str, Any]:
    """
    Intelligently resolve a partial file path to a full Unity project path.
    
    Uses multiple strategies:
    1. Direct path (if already complete with Assets/ and extension)
    2. SQLite database query (for indexed files)
    3. Unity structure heuristics (guess common locations)
    4. Filesystem search (last resort)
    
    Returns:
        {
            "success": bool,
            "resolved_path": str or None,
            "candidates": List[str],  # If multiple matches
            "strategy_used": str,
            "needs_clarification": bool
        }
    """
    logger.info(f"🔍 [PathResolver] Resolving: '{partial_path}'")
    logger.info(f"🔍 [PathResolver]   Operation: {operation}")
    logger.info(f"🔍 [PathResolver]   Project root: {project_root}")
    
    # STRATEGY 1: Check if already complete path
    if _is_complete_path(partial_path):
        logger.info(f"🔍 [PathResolver] Strategy 1: Path is already complete")
        abs_path = project_root / partial_path
        if abs_path.exists():
            return {
                "success": True,
                "resolved_path": partial_path,
                "candidates": [partial_path],
                "strategy_used": "direct",
                "needs_clarification": False
            }
    
    # STRATEGY 2: Query SQLite database (FASTEST and MOST ACCURATE)
    if sqlite_path and sqlite_path.exists():
        logger.info(f"🔍 [PathResolver] Strategy 2: Querying movesia.db")
        db_result = await asyncio.to_thread(
            _query_database_for_file, 
            partial_path, 
            project_root, 
            sqlite_path, 
            operation
        )
        if db_result["success"]:
            return db_result
    
    # STRATEGY 3: Unity structure heuristics (common locations)
    logger.info(f"🔍 [PathResolver] Strategy 3: Unity structure heuristics")
    heuristic_result = _apply_unity_heuristics(partial_path, project_root, operation)
    if heuristic_result["success"]:
        return heuristic_result
    
    # STRATEGY 4: Filesystem search (last resort, slower)
    logger.info(f"🔍 [PathResolver] Strategy 4: Filesystem search")
    search_result = await asyncio.to_thread(
        _search_filesystem,
        partial_path,
        project_root
    )
    if search_result["success"]:
        return search_result
    
    # ALL STRATEGIES FAILED
    logger.warning(f"🔍 [PathResolver] ❌ All strategies failed")
    return {
        "success": False,
        "resolved_path": None,
        "candidates": [],
        "strategy_used": "none",
        "needs_clarification": False,
        "error": f"Could not find file matching '{partial_path}' in project"
    }


def _is_complete_path(path: str) -> bool:
    """Check if path is already complete (has Assets/ and extension)."""
    path_lower = path.lower()
    has_assets_prefix = path_lower.startswith("assets/") or path_lower.startswith("assets\\")
    has_extension = any(path_lower.endswith(ext) for ext in [".cs", ".unity", ".prefab", ".asset", ".mat", ".shader", ".json", ".txt"])
    return has_assets_prefix and has_extension


def _query_database_for_file(
    filename: str,
    project_root: Path,
    sqlite_path: Path,
    operation: str
) -> Dict[str, Any]:
    """Query movesia.db for matching files using fuzzy search."""
    import sqlite3
    
    try:
        conn = sqlite3.connect(str(sqlite_path), timeout=5.0)
        cursor = conn.cursor()
        
        # Clean filename for search
        clean_name = filename.strip().lower()
        # Remove common prefixes if user included them
        clean_name = clean_name.replace("assets/", "").replace("assets\\", "")
        
        logger.info(f"🔍 [Database] Searching for: {clean_name}")
        
        # FUZZY SEARCH QUERY - Updated for 'assets' table structure
        query = """
            SELECT path, kind, size 
            FROM assets 
            WHERE deleted = 0 
              AND (LOWER(path) LIKE ? OR LOWER(path) LIKE ?)
            ORDER BY 
                CASE 
                    WHEN LOWER(path) LIKE ? THEN 1          -- Exact filename match (highest priority)
                    WHEN LOWER(path) LIKE ? THEN 2          -- Path ends with term
                    WHEN LOWER(path) LIKE ? THEN 3          -- Path contains term
                    ELSE 4
                END,
                LENGTH(path) ASC                             -- Prefer shorter paths
            LIMIT 10
        """
        
        # Wildcards for fuzzy matching
        # Match: "Assets/Scripts/test5279.cs" when searching "test5279"
        exact_filename = f"%/{clean_name}.cs"      # Match exact filename
        fuzzy_pattern = f"%{clean_name}%"          # Match anywhere in path
        
        cursor.execute(query, (
            exact_filename,      # Exact filename match with .cs
            fuzzy_pattern,       # Fuzzy match anywhere
            exact_filename,      # ORDER BY: exact filename (priority 1)
            exact_filename,      # ORDER BY: ends with term (priority 2)
            fuzzy_pattern,       # ORDER BY: contains term (priority 3)
        ))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            logger.info(f"🔍 [Database] No results found")
            return {"success": False}
        
        logger.info(f"🔍 [Database] Found {len(results)} candidates")
        
        candidates = []
        for row in results:
            path, kind, size = row
            # Extract filename from path for display
            name = path.split("/")[-1] if "/" in path else path
            
            candidates.append({
                "path": path,
                "name": name,
                "kind": kind,
                "size": size
            })
            logger.info(f"🔍 [Database]   - {path} ({kind}, {size} bytes)")
        
        # EXACT MATCH?
        # Check if any candidate's filename exactly matches (case-insensitive)
        exact_matches = [
            c for c in candidates 
            if c["name"].lower() == clean_name.lower() or 
               c["name"].lower() == f"{clean_name}.cs".lower()
        ]
        
        if len(exact_matches) == 1:
            # Single exact match - use it!
            chosen = exact_matches[0]
            logger.info(f"🔍 [Database] ✅ Exact match found: {chosen['path']}")
            return {
                "success": True,
                "resolved_path": chosen["path"],
                "candidates": [c["path"] for c in candidates],
                "strategy_used": "database_exact",
                "needs_clarification": False
            }
        
        elif len(results) == 1:
            # Single fuzzy match - use it!
            chosen = candidates[0]
            logger.info(f"🔍 [Database] ✅ Single fuzzy match: {chosen['path']}")
            return {
                "success": True,
                "resolved_path": chosen["path"],
                "candidates": [c["path"] for c in candidates],
                "strategy_used": "database_fuzzy",
                "needs_clarification": False
            }
        
        else:
            # Multiple matches - need clarification
            logger.info(f"🔍 [Database] ⚠️ Multiple matches, needs clarification")
            return {
                "success": False,
                "resolved_path": None,
                "candidates": [c["path"] for c in candidates],
                "strategy_used": "database_multiple",
                "needs_clarification": True,
                "error": f"Multiple files match '{filename}'. Please be more specific."
            }
    
    except Exception as e:
        logger.error(f"🔍 [Database] Error: {str(e)}")
        return {"success": False}


def _apply_unity_heuristics(
    filename: str,
    project_root: Path,
    operation: str
) -> Dict[str, Any]:
    """Apply Unity project structure heuristics to guess file location."""
    # Clean filename
    clean_name = filename.strip()
    if not any(clean_name.endswith(ext) for ext in [".cs", ".unity", ".prefab", ".asset"]):
        clean_name += ".cs"  # Default to .cs for scripts
    
    # Common Unity locations by file type
    search_locations = []
    
    if clean_name.endswith(".cs"):
        search_locations = [
            f"Assets/Scripts/{clean_name}",
            f"Assets/{clean_name}",
            f"Assets/Scripts/Player/{clean_name}",
            f"Assets/Scripts/UI/{clean_name}",
            f"Assets/Scripts/Game/{clean_name}",
        ]
    elif clean_name.endswith(".prefab"):
        search_locations = [
            f"Assets/Prefabs/{clean_name}",
            f"Assets/{clean_name}",
        ]
    elif clean_name.endswith(".unity"):
        search_locations = [
            f"Assets/Scenes/{clean_name}",
            f"Assets/{clean_name}",
        ]
    else:
        search_locations = [f"Assets/{clean_name}"]
    
    logger.info(f"🔍 [Heuristics] Checking {len(search_locations)} locations")
    
    found_paths = []
    for rel_path in search_locations:
        abs_path = project_root / rel_path
        if abs_path.exists():
            logger.info(f"🔍 [Heuristics]   ✅ Found: {rel_path}")
            found_paths.append(rel_path)
        else:
            logger.debug(f"🔍 [Heuristics]   ❌ Not found: {rel_path}")
    
    if len(found_paths) == 1:
        return {
            "success": True,
            "resolved_path": found_paths[0],
            "candidates": found_paths,
            "strategy_used": "heuristics",
            "needs_clarification": False
        }
    elif len(found_paths) > 1:
        return {
            "success": False,
            "resolved_path": None,
            "candidates": found_paths,
            "strategy_used": "heuristics_multiple",
            "needs_clarification": True,
            "error": f"Multiple files match '{filename}' in common locations"
        }
    
    return {"success": False}


def _search_filesystem(
    filename: str,
    project_root: Path
) -> Dict[str, Any]:
    """Search filesystem for matching files (slow but thorough)."""
    import glob
    
    clean_name = filename.strip().lower()
    assets_dir = project_root / "Assets"
    
    if not assets_dir.exists():
        return {"success": False}
    
    logger.info(f"🔍 [Filesystem] Searching in: {assets_dir}")
    
    # Search patterns
    patterns = [
        f"**/*{clean_name}*",  # Anywhere in name
        f"**/{clean_name}",    # Exact name
        f"**/{clean_name}.cs", # With .cs extension
    ]
    
    found_paths = []
    for pattern in patterns:
        matches = list(assets_dir.glob(pattern))
        for match in matches:
            if match.is_file():
                rel_path = match.relative_to(project_root).as_posix()
                if rel_path not in found_paths:
                    found_paths.append(rel_path)
                    logger.info(f"🔍 [Filesystem]   Found: {rel_path}")
    
    if len(found_paths) == 1:
        return {
            "success": True,
            "resolved_path": found_paths[0],
            "candidates": found_paths,
            "strategy_used": "filesystem",
            "needs_clarification": False
        }
    elif len(found_paths) > 1:
        return {
            "success": False,
            "resolved_path": None,
            "candidates": found_paths[:10],  # Limit to 10
            "strategy_used": "filesystem_multiple",
            "needs_clarification": True,
            "error": f"Found {len(found_paths)} files matching '{filename}'"
        }
    
    return {"success": False}


# ============================================================================
# FILE OPERATION TOOLS (5 separate tools)
# ============================================================================

@tool
async def read_file(
    file_path: str,
    config: RunnableConfig,
) -> Dict[str, Any]:
    """Read file contents from the Unity project.
    
    Args:
        file_path: Target file path (can be partial like "test4073" or full like "Assets/Scripts/test4073.cs")
        
    Returns:
        File contents and metadata
    """
    configurable = config.get("configurable", {})
    project_root = configurable.get("project_root")
    sqlite_path = configurable.get("sqlite_path")
    
    if not project_root:
        return {"success": False, "error": "Project root not available"}
    
    project_root_path = Path(project_root)
    sqlite_db_path = Path(sqlite_path) if sqlite_path else None
    
    resolution = await _resolve_file_path(
        partial_path=file_path,
        project_root=project_root_path,
        sqlite_path=sqlite_db_path,
        operation="read"
    )
    
    if resolution.get("needs_clarification"):
        return {
            "success": False,
            "error": resolution.get("error", "Multiple files found"),
            "candidates": resolution.get("candidates", [])[:5]
        }
    
    if not resolution.get("success"):
        return {
            "success": False,
            "error": f"Could not find file '{file_path}'"
        }
    
    resolved_path = resolution["resolved_path"]
    abs_path = project_root_path / resolved_path
    
    return await _file_read(abs_path, resolved_path)


@tool
async def write_file(
    file_path: str,
    content: str,
    config: RunnableConfig,
) -> Dict[str, Any]:
    """Write or create a file in the Unity project (requires approval).
    
    Args:
        file_path: Target file path
        content: Content to write
        
    Returns:
        Approval request for destructive operation
    """
    configurable = config.get("configurable", {})
    project_root = configurable.get("project_root")
    sqlite_path = configurable.get("sqlite_path")
    
    if not project_root:
        return {"success": False, "error": "Project root not available"}
    
    project_root_path = Path(project_root)
    sqlite_db_path = Path(sqlite_path) if sqlite_path else None
    
    resolution = await _resolve_file_path(
        partial_path=file_path,
        project_root=project_root_path,
        sqlite_path=sqlite_db_path,
        operation="write"
    )
    
    if not resolution.get("success"):
        resolved_path = file_path
        if not resolved_path.startswith("Assets"):
            resolved_path = f"Assets/Scripts/{resolved_path}"
        if not any(resolved_path.endswith(ext) for ext in [".cs", ".txt", ".json"]):
            resolved_path += ".cs"
    else:
        resolved_path = resolution["resolved_path"]
    
    abs_path = project_root_path / resolved_path
    return await _file_write_prepare(abs_path, resolved_path, content)


@tool
async def modify_file(
    file_path: str,
    modification_spec: Dict[str, Any],
    config: RunnableConfig,
) -> Dict[str, Any]:
    """Modify an existing file in the Unity project (requires approval).
    
    Args:
        file_path: Target file path
        modification_spec: Specification for surgical edits
        
    Returns:
        Approval request for destructive operation
    """
    configurable = config.get("configurable", {})
    project_root = configurable.get("project_root")
    sqlite_path = configurable.get("sqlite_path")
    
    if not project_root:
        return {"success": False, "error": "Project root not available"}
    
    project_root_path = Path(project_root)
    sqlite_db_path = Path(sqlite_path) if sqlite_path else None
    
    resolution = await _resolve_file_path(
        partial_path=file_path,
        project_root=project_root_path,
        sqlite_path=sqlite_db_path,
        operation="modify"
    )
    
    if not resolution.get("success"):
        return {"success": False, "error": f"Could not find file '{file_path}'"}
    
    resolved_path = resolution["resolved_path"]
    abs_path = project_root_path / resolved_path
    
    return await _file_modify_prepare(abs_path, resolved_path, modification_spec)


@tool
async def delete_file(
    file_path: str,
    config: RunnableConfig,
) -> Dict[str, Any]:
    """Delete a file from the Unity project (requires approval).
    
    Args:
        file_path: Target file path
        
    Returns:
        Approval request for destructive operation
    """
    configurable = config.get("configurable", {})
    project_root = configurable.get("project_root")
    sqlite_path = configurable.get("sqlite_path")
    
    if not project_root:
        return {"success": False, "error": "Project root not available"}
    
    project_root_path = Path(project_root)
    sqlite_db_path = Path(sqlite_path) if sqlite_path else None
    
    resolution = await _resolve_file_path(
        partial_path=file_path,
        project_root=project_root_path,
        sqlite_path=sqlite_db_path,
        operation="delete"
    )
    
    if not resolution.get("success"):
        return {"success": False, "error": f"Could not find file '{file_path}'"}
    
    resolved_path = resolution["resolved_path"]
    abs_path = project_root_path / resolved_path
    
    return await _file_delete_prepare(abs_path, resolved_path)


@tool
async def move_file(
    file_path: str,
    new_path: str,
    config: RunnableConfig,
) -> Dict[str, Any]:
    """Move a file to a new location in the Unity project (requires approval).
    
    Args:
        file_path: Source file path
        new_path: Destination path
        
    Returns:
        Approval request for destructive operation
    """
    configurable = config.get("configurable", {})
    project_root = configurable.get("project_root")
    sqlite_path = configurable.get("sqlite_path")
    
    if not project_root:
        return {"success": False, "error": "Project root not available"}
    
    project_root_path = Path(project_root)
    sqlite_db_path = Path(sqlite_path) if sqlite_path else None
    
    resolution = await _resolve_file_path(
        partial_path=file_path,
        project_root=project_root_path,
        sqlite_path=sqlite_db_path,
        operation="move"
    )
    
    if not resolution.get("success"):
        return {"success": False, "error": f"Could not find file '{file_path}'"}
    
    resolved_path = resolution["resolved_path"]
    abs_path = project_root_path / resolved_path
    new_abs_path = project_root_path / new_path
    
    return await _file_move_prepare(abs_path, new_abs_path, resolved_path, new_path)


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
    logger.info(f"🗑️ [FileOp] Preparing delete operation for: {rel_path}")
    logger.info(f"🗑️ [FileOp]   Absolute path: {abs_path}")
    
    logger.info(f"🗑️ [FileOp] Checking if file exists...")
    exists = await asyncio.to_thread(abs_path.exists)
    logger.info(f"🗑️ [FileOp]   Exists: {exists}")
    
    if not exists:
        logger.warning(f"🗑️ [FileOp] ❌ File not found: {rel_path}")
        return {"success": False, "error": f"File not found: {rel_path}"}
    
    # ✅ ADD: Check if it's actually a file
    logger.info(f"🗑️ [FileOp] Checking if path is a file...")
    is_file = await asyncio.to_thread(abs_path.is_file)
    logger.info(f"🗑️ [FileOp]   Is file: {is_file}")
    
    if not is_file:
        logger.warning(f"🗑️ [FileOp] ❌ Path is not a file (might be a directory): {rel_path}")
        return {"success": False, "error": f"Path is not a file (might be a directory): {rel_path}"}
    
    logger.info(f"🗑️ [FileOp] Reading file content for preview...")
    try:
        content = await asyncio.to_thread(abs_path.read_text, encoding="utf-8")
        logger.info(f"🗑️ [FileOp]   Content read successfully ({len(content)} chars)")
    except Exception as e:
        logger.warning(f"🗑️ [FileOp]   Could not read content: {type(e).__name__}: {str(e)}")
        content = "[Binary file or unreadable content]"
    
    logger.info(f"🗑️ [FileOp] ✅ Returning approval request for: {rel_path}")
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
    
    logger.info(f"📝 [FileOp] Starting write operation for: {rel_path}")
    logger.info(f"📝 [FileOp]   Absolute path: {abs_path}")
    logger.info(f"📝 [FileOp]   File already exists: {file_exists}")
    logger.info(f"📝 [FileOp]   Content size: {len(content)} chars")
    
    try:
        # ✅ ADD: Ensure parent directory exists
        logger.info(f"📝 [FileOp] Checking parent directory...")
        logger.info(f"📝 [FileOp]   Parent: {abs_path.parent}")
        
        parent_exists = await asyncio.to_thread(abs_path.parent.exists)
        logger.info(f"📝 [FileOp]   Parent exists: {parent_exists}")
        
        if not parent_exists:
            logger.info(f"📝 [FileOp] Creating parent directory...")
            await asyncio.to_thread(abs_path.parent.mkdir, parents=True, exist_ok=True)
            logger.info(f"📝 [FileOp]   ✅ Parent directory created")
        
        logger.info(f"📝 [FileOp] Writing content to file...")
        await asyncio.to_thread(abs_path.write_text, content, encoding="utf-8")
        logger.info(f"📝 [FileOp]   Write call completed")
        
        # ✅ ADD: Verify write succeeded
        logger.info(f"📝 [FileOp] Verifying write...")
        verify_exists = await asyncio.to_thread(abs_path.exists)
        logger.info(f"📝 [FileOp]   File exists after write: {verify_exists}")
        
        if not verify_exists:
            logger.error(f"📝 [FileOp] ❌ File was not created: {rel_path}")
            return {"success": False, "error": f"File was not created: {rel_path}"}
        
        logger.info(f"📝 [FileOp] ✅ File written successfully: {rel_path}")
        return {
            "success": True,
            "operation": "write",
            "file_path": rel_path,
            "created": not file_exists,
            "size_bytes": len(content.encode("utf-8")),
            "line_count": len(content.split("\n"))
        }
        
    except PermissionError as e:
        logger.error(f"📝 [FileOp] ❌ Permission denied: {rel_path}")
        logger.error(f"📝 [FileOp]   Error details: {str(e)}")
        return {"success": False, "error": f"Permission denied writing to: {rel_path}"}
    except OSError as e:
        logger.error(f"📝 [FileOp] ❌ OS error: {rel_path}")
        logger.error(f"📝 [FileOp]   Error type: {type(e).__name__}")
        logger.error(f"📝 [FileOp]   Error details: {str(e)}")
        return {"success": False, "error": f"OS error writing file: {str(e)}"}
    except Exception as e:
        logger.error(f"📝 [FileOp] ❌ Unexpected error: {rel_path}")
        logger.error(f"📝 [FileOp]   Error type: {type(e).__name__}")
        logger.error(f"📝 [FileOp]   Error details: {str(e)}")
        logger.exception(f"📝 [FileOp]   Full traceback:")
        return {"success": False, "error": f"Failed to write file: {type(e).__name__}: {str(e)}"}


async def _execute_modify(pending_op: Dict[str, Any]) -> Dict[str, Any]:
    """Execute modify operation."""
    abs_path = Path(pending_op["abs_path"])
    rel_path = pending_op["rel_path"]
    content = pending_op["content"]
    
    logger.info(f"✏️ [FileOp] Starting modify operation for: {rel_path}")
    logger.info(f"✏️ [FileOp]   Absolute path: {abs_path}")
    logger.info(f"✏️ [FileOp]   New content size: {len(content)} chars")
    
    try:
        logger.info(f"✏️ [FileOp] Checking if file exists...")
        exists = await asyncio.to_thread(abs_path.exists)
        logger.info(f"✏️ [FileOp]   Exists: {exists}")
        
        if not exists:
            logger.error(f"✏️ [FileOp] ❌ File does not exist: {rel_path}")
            return {"success": False, "error": f"File not found: {rel_path}"}
        
        logger.info(f"✏️ [FileOp] Writing modified content...")
        await asyncio.to_thread(abs_path.write_text, content, encoding="utf-8")
        logger.info(f"✏️ [FileOp]   Write completed")
        
        logger.info(f"✏️ [FileOp] ✅ File modified successfully: {rel_path}")
        return {
            "success": True,
            "operation": "modify",
            "file_path": rel_path
        }
        
    except PermissionError as e:
        logger.error(f"✏️ [FileOp] ❌ Permission denied: {rel_path}")
        logger.error(f"✏️ [FileOp]   Error details: {str(e)}")
        return {"success": False, "error": f"Permission denied modifying: {rel_path}"}
    except Exception as e:
        logger.error(f"✏️ [FileOp] ❌ Error modifying file: {rel_path}")
        logger.error(f"✏️ [FileOp]   Error type: {type(e).__name__}")
        logger.error(f"✏️ [FileOp]   Error details: {str(e)}")
        logger.exception(f"✏️ [FileOp]   Full traceback:")
        return {"success": False, "error": f"Failed to modify file: {str(e)}"}


async def _execute_delete(pending_op: Dict[str, Any]) -> Dict[str, Any]:
    """Execute delete operation."""
    abs_path = Path(pending_op["abs_path"])
    rel_path = pending_op["rel_path"]
    
    logger.info(f"🗑️ [FileOp] Starting delete operation for: {rel_path}")
    logger.info(f"🗑️ [FileOp]   Absolute path: {abs_path}")
    
    try:
        # ✅ FIX 1: Verify file still exists before deletion
        logger.info(f"🗑️ [FileOp] Checking if file exists...")
        exists = await asyncio.to_thread(abs_path.exists)
        logger.info(f"🗑️ [FileOp]   File exists: {exists}")
        
        if not exists:
            logger.warning(f"🗑️ [FileOp] ❌ File no longer exists: {rel_path}")
            return {
                "success": False,
                "error": f"File no longer exists: {rel_path}",
                "operation": "delete"
            }
        
        # ✅ FIX 2: Check if it's a file (not a directory)
        logger.info(f"🗑️ [FileOp] Checking if path is a file...")
        is_file = await asyncio.to_thread(abs_path.is_file)
        logger.info(f"🗑️ [FileOp]   Is file: {is_file}")
        
        if not is_file:
            logger.warning(f"🗑️ [FileOp] ❌ Path is not a file: {rel_path}")
            return {
                "success": False,
                "error": f"Path is not a file: {rel_path}",
                "operation": "delete"
            }
        
        # ✅ FIX 3: Delete with proper async handling
        logger.info(f"🗑️ [FileOp] Attempting to delete file...")
        await asyncio.to_thread(abs_path.unlink, missing_ok=False)
        logger.info(f"🗑️ [FileOp]   Unlink call completed")
        
        # ✅ FIX 4: Verify deletion succeeded
        logger.info(f"🗑️ [FileOp] Verifying deletion...")
        still_exists = await asyncio.to_thread(abs_path.exists)
        logger.info(f"🗑️ [FileOp]   File still exists after deletion: {still_exists}")
        
        if still_exists:
            logger.error(f"🗑️ [FileOp] ❌ File still exists after deletion attempt: {rel_path}")
            return {
                "success": False,
                "error": f"File still exists after deletion attempt: {rel_path}",
                "operation": "delete"
            }
        
        logger.info(f"🗑️ [FileOp] ✅ File deleted successfully: {rel_path}")
        return {
            "success": True,
            "operation": "delete",
            "file_path": rel_path,
            "deleted": True
        }
        
    except PermissionError as e:
        logger.error(f"🗑️ [FileOp] ❌ Permission denied: {rel_path}")
        logger.error(f"🗑️ [FileOp]   Error details: {str(e)}")
        return {
            "success": False,
            "error": f"Permission denied deleting file: {rel_path}",
            "operation": "delete"
        }
    except OSError as e:
        logger.error(f"🗑️ [FileOp] ❌ OS error: {rel_path}")
        logger.error(f"🗑️ [FileOp]   Error type: {type(e).__name__}")
        logger.error(f"🗑️ [FileOp]   Error details: {str(e)}")
        return {
            "success": False,
            "error": f"OS error deleting file {rel_path}: {str(e)}",
            "operation": "delete"
        }
    except Exception as e:
        logger.error(f"🗑️ [FileOp] ❌ Unexpected error: {rel_path}")
        logger.error(f"🗑️ [FileOp]   Error type: {type(e).__name__}")
        logger.error(f"🗑️ [FileOp]   Error details: {str(e)}")
        logger.exception(f"🗑️ [FileOp]   Full traceback:")
        return {
            "success": False,
            "error": f"Unexpected error deleting file {rel_path}: {type(e).__name__}: {str(e)}",
            "operation": "delete"
        }


async def _execute_move(pending_op: Dict[str, Any]) -> Dict[str, Any]:
    """Execute move operation."""
    abs_path = Path(pending_op["abs_path"])
    new_abs_path = Path(pending_op["new_abs_path"])
    from_path = pending_op["from_path"]
    to_path = pending_op["to_path"]
    
    logger.info(f"📦 [FileOp] Starting move operation")
    logger.info(f"📦 [FileOp]   From: {from_path}")
    logger.info(f"📦 [FileOp]   To: {to_path}")
    logger.info(f"📦 [FileOp]   Absolute from: {abs_path}")
    logger.info(f"📦 [FileOp]   Absolute to: {new_abs_path}")
    
    try:
        logger.info(f"📦 [FileOp] Checking source file...")
        exists = await asyncio.to_thread(abs_path.exists)
        logger.info(f"📦 [FileOp]   Source exists: {exists}")
        
        if not exists:
            logger.error(f"📦 [FileOp] ❌ Source file not found: {from_path}")
            return {"success": False, "error": f"Source file not found: {from_path}"}
        
        logger.info(f"📦 [FileOp] Checking destination...")
        dest_exists = await asyncio.to_thread(new_abs_path.exists)
        logger.info(f"📦 [FileOp]   Destination exists: {dest_exists}")
        
        if dest_exists:
            logger.error(f"📦 [FileOp] ❌ Destination already exists: {to_path}")
            return {"success": False, "error": f"Destination already exists: {to_path}"}
        
        logger.info(f"📦 [FileOp] Creating destination parent directory if needed...")
        await asyncio.to_thread(new_abs_path.parent.mkdir, parents=True, exist_ok=True)
        logger.info(f"📦 [FileOp]   Parent directory ready")
        
        logger.info(f"📦 [FileOp] Moving file...")
        await asyncio.to_thread(abs_path.rename, new_abs_path)
        logger.info(f"📦 [FileOp]   Move completed")
        
        logger.info(f"📦 [FileOp] Verifying move...")
        old_still_exists = await asyncio.to_thread(abs_path.exists)
        new_exists = await asyncio.to_thread(new_abs_path.exists)
        logger.info(f"📦 [FileOp]   Old location exists: {old_still_exists}")
        logger.info(f"📦 [FileOp]   New location exists: {new_exists}")
        
        if old_still_exists or not new_exists:
            logger.error(f"📦 [FileOp] ❌ Move verification failed")
            return {"success": False, "error": f"Move verification failed"}
        
        logger.info(f"📦 [FileOp] ✅ File moved successfully")
        return {
            "success": True,
            "operation": "move",
            "from_path": from_path,
            "to_path": to_path,
            "moved": True
        }
        
    except PermissionError as e:
        logger.error(f"📦 [FileOp] ❌ Permission denied")
        logger.error(f"📦 [FileOp]   Error details: {str(e)}")
        return {"success": False, "error": f"Permission denied moving file: {str(e)}"}
    except Exception as e:
        logger.error(f"📦 [FileOp] ❌ Error moving file")
        logger.error(f"📦 [FileOp]   Error type: {type(e).__name__}")
        logger.error(f"📦 [FileOp]   Error details: {str(e)}")
        logger.exception(f"📦 [FileOp]   Full traceback:")
        return {"success": False, "error": f"Failed to move file: {str(e)}"}