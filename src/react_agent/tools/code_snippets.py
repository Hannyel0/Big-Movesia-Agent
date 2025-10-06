"""Two-stage hybrid search: exact file name matching + semantic search."""

from __future__ import annotations
from typing import Optional, Dict, Any, List
import os
import re
from datetime import datetime, UTC

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from react_agent.context import Context


EMBED_SERVER_URL = os.getenv("EMBED_SERVER_URL", "http://127.0.0.1:8766")
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "movesia")


def _extract_code_signature(code: str, max_lines: int = 20) -> str:
    """Extract meaningful code signature without full implementation.
    
    Returns: class definition, public methods, properties - no method bodies.
    """
    lines = code.split('\n')
    signature_lines = []
    in_method = False
    brace_count = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Always include: using, namespace, class declarations
        if any(stripped.startswith(kw) for kw in ['using ', 'namespace ', 'public class ', 'class ']):
            signature_lines.append(line)
            if '{' in line:
                brace_count += line.count('{') - line.count('}')
            continue
        
        # Include public/private method signatures but not bodies
        if any(stripped.startswith(kw) for kw in ['public ', 'private ', 'protected ', 'internal ']):
            # Check if it's a method/property declaration
            if '(' in stripped or 'get;' in stripped or 'set;' in stripped or '=>' in stripped:
                signature_lines.append(line)
                if '{' in stripped and '}' not in stripped:
                    # Method with body - skip the body
                    in_method = True
                    brace_count = stripped.count('{')
                continue
        
        # Track braces to skip method bodies
        if in_method:
            brace_count += line.count('{') - line.count('}')
            if brace_count <= 0:
                in_method = False
                signature_lines.append(line.split('}')[0] + '}  // ... implementation omitted')
            continue
        
        # Include comments and attributes
        if stripped.startswith('//') or stripped.startswith('[') or stripped.startswith('///'):
            signature_lines.append(line)
            continue
        
        # Stop if we have enough
        if len(signature_lines) >= max_lines:
            break
    
    if len(signature_lines) < len(lines):
        signature_lines.append('\n// ... additional code omitted for brevity ...')
    
    return '\n'.join(signature_lines)


def _extract_metadata_from_code(code: str) -> Dict[str, Any]:
    """Extract rich metadata from code without including full code."""
    metadata = {
        "classes": [],
        "public_methods": [],
        "properties": [],
        "unity_callbacks": [],
        "namespaces": [],
        "using_directives": []
    }
    
    lines = code.split('\n')
    current_class = None
    
    for line in lines:
        stripped = line.strip()
        
        # Extract using directives
        if stripped.startswith('using '):
            using = stripped.replace('using ', '').replace(';', '').strip()
            metadata["using_directives"].append(using)
        
        # Extract namespace
        if stripped.startswith('namespace '):
            namespace = stripped.replace('namespace ', '').replace('{', '').strip()
            metadata["namespaces"].append(namespace)
        
        # Extract class names
        if 'class ' in stripped and not stripped.startswith('//'):
            match = re.search(r'class\s+(\w+)', stripped)
            if match:
                current_class = match.group(1)
                metadata["classes"].append(current_class)
        
        # Extract public methods
        if current_class and stripped.startswith('public ') and '(' in stripped:
            # Extract method signature
            match = re.search(r'public\s+\w+\s+(\w+)\s*\(', stripped)
            if match:
                method_name = match.group(1)
                metadata["public_methods"].append(method_name)
                
                # Check if it's a Unity callback
                if method_name in ['Start', 'Update', 'Awake', 'FixedUpdate', 'OnEnable', 
                                   'OnDisable', 'OnDestroy', 'LateUpdate', 'OnGUI']:
                    metadata["unity_callbacks"].append(method_name)
        
        # Extract properties
        if current_class and stripped.startswith('public ') and ('{ get;' in stripped or '{ set;' in stripped):
            match = re.search(r'public\s+\w+\s+(\w+)\s*{', stripped)
            if match:
                metadata["properties"].append(match.group(1))
    
    return metadata


def _create_token_efficient_snippet(
    payload: Dict[str, Any],
    score: float,
    match_type: str,
    include_full_code: bool = False,
    max_code_chars: int = 2000
) -> Dict[str, Any]:
    """Create snippet with token usage in mind.
    
    Args:
        include_full_code: If False, only include code signature/metadata
        max_code_chars: Maximum characters of code to include when full code requested
    """
    code_text = payload.get("text", "")
    file_path = payload.get("rel_path", "unknown")
    
    # Extract metadata regardless
    metadata = _extract_metadata_from_code(code_text)
    
    snippet = {
        "file_path": file_path,
        "file_name": file_path.split('/')[-1],
        "line_range": payload.get("range", ""),
        "relevance_score": round(score, 3),
        "match_type": match_type,
        "file_hash": payload.get("file_hash", ""),
        
        # Rich metadata (low token cost)
        "classes": metadata["classes"],
        "public_methods": metadata["public_methods"],
        "properties": metadata["properties"],
        "unity_callbacks": metadata["unity_callbacks"],
        "namespaces": metadata["namespaces"],
        "using_directives": metadata["using_directives"][:5],  # Limit to 5
        
        # Code statistics
        "total_lines": len(code_text.split('\n')),
        "code_size_bytes": len(code_text),
    }
    
    if include_full_code:
        # Truncate if too long
        if len(code_text) > max_code_chars:
            truncated_code = code_text[:max_code_chars]
            # Try to end at a complete line
            last_newline = truncated_code.rfind('\n')
            if last_newline > max_code_chars * 0.8:  # If we can get 80%+ with clean line break
                truncated_code = truncated_code[:last_newline]
            
            snippet["code"] = truncated_code + "\n\n// ... code truncated ..."
            snippet["code_truncated"] = True
            snippet["truncation_point"] = max_code_chars
        else:
            snippet["code"] = code_text
            snippet["code_truncated"] = False
    else:
        # Just include signature
        snippet["code_signature"] = _extract_code_signature(code_text, max_lines=15)
        snippet["full_code_available"] = True
    
    return snippet


def _extract_file_name_patterns(query: str) -> List[str]:
    """Extract all possible file name patterns from query.
    
    Returns multiple patterns to try for fuzzy matching.
    
    Examples:
        "Movesia events script" -> ["MovesiaEvents", "Movesia", "Events"]
        "PlayerController" -> ["PlayerController", "Player", "Controller"]
    """
    patterns = []
    
    # Check for explicit .cs file
    cs_match = re.search(r'(\w+)\.cs\b', query, re.IGNORECASE)
    if cs_match:
        patterns.append(cs_match.group(1))
    
    # Remove noise words
    noise_words = {
        'the', 'a', 'an', 'get', 'me', 'find', 'show', 'script', 'code',
        'file', 'class', 'my', 'your', 'can', 'you', 'please', 'want',
        'need', 'looking', 'for', 'search', 'in', 'from', 'project'
    }
    
    words = [w for w in query.split() if w.lower() not in noise_words]
    
    if not words:
        return patterns
    
    # Pattern 1: CamelCase combination of all words
    if len(words) >= 2:
        camel_case = ''.join(w.capitalize() for w in words)
        patterns.append(camel_case)
    
    # Pattern 2: Individual significant words (capitalized)
    for word in words:
        if len(word) > 2:  # Skip very short words
            patterns.append(word.capitalize())
    
    # Pattern 3: Two-word combinations
    if len(words) >= 2:
        for i in range(len(words) - 1):
            combo = words[i].capitalize() + words[i + 1].capitalize()
            patterns.append(combo)
    
    return list(set(patterns))  # Remove duplicates


async def _search_by_file_name(
    project_id: str,
    file_patterns: List[str],
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Search Qdrant by file path using scroll/filter (no vectors needed).
    
    This finds files by exact path matching, bypassing semantic search entirely.
    """
    import httpx
    
    if not file_patterns:
        return []
    
    matched_results = []
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Use Qdrant scroll to get all script files for this project
        # Then filter by file name in Python (Qdrant doesn't support regex in filters)
        scroll_payload = {
            "filter": {
                "must": [
                    {"key": "project_id", "match": {"value": project_id}},
                    {"key": "kind", "match": {"value": "Script"}}
                ]
            },
            "limit": 100,  # Get up to 100 scripts to search through
            "with_payload": True,
            "with_vector": False  # Don't need vectors for name matching
        }
        
        response = await client.post(
            f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/scroll",
            json=scroll_payload,
            timeout=30.0
        )
        
        if response.status_code != 200:
            return []
        
        scroll_results = response.json()
        points = scroll_results.get("result", {}).get("points", [])
        
        # Filter results by file name patterns
        for point in points:
            payload = point.get("payload", {})
            rel_path = payload.get("rel_path", "")
            file_name = rel_path.split('/')[-1].replace('.cs', '')
            
            # Check if any pattern matches this file name
            for pattern in file_patterns:
                pattern_lower = pattern.lower()
                file_lower = file_name.lower()
                
                # Exact match
                if pattern_lower == file_lower:
                    matched_results.append({
                        "payload": payload,
                        "score": 1.0,  # Perfect match
                        "match_type": "exact",
                        "pattern": pattern
                    })
                    break
                
                # Contains match
                elif pattern_lower in file_lower or file_lower in pattern_lower:
                    matched_results.append({
                        "payload": payload,
                        "score": 0.9,  # High match
                        "match_type": "contains",
                        "pattern": pattern
                    })
                    break
                
                # Starts with match
                elif file_lower.startswith(pattern_lower):
                    matched_results.append({
                        "payload": payload,
                        "score": 0.85,
                        "match_type": "starts_with",
                        "pattern": pattern
                    })
                    break
    
    # Sort by score and limit
    matched_results.sort(key=lambda x: x["score"], reverse=True)
    return matched_results[:limit]


async def _search_by_semantic(
    project_id: str,
    query: str,
    limit: int = 10,
    score_threshold: float = 0.3
) -> List[Dict[str, Any]]:
    """Standard semantic vector search."""
    import httpx
    
    # Generate embedding
    embedding_result = await _get_local_embedding(query)
    query_vector = embedding_result["embedding"]
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        search_payload = {
            "vector": query_vector,
            "limit": limit,
            "with_payload": True,
            "score_threshold": score_threshold,
            "filter": {
                "must": [
                    {"key": "project_id", "match": {"value": project_id}},
                    {"key": "kind", "match": {"value": "Script"}}
                ]
            }
        }
        
        response = await client.post(
            f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/search",
            json=search_payload,
            timeout=30.0
        )
        
        if response.status_code != 200:
            return []
        
        search_results = response.json()
        
        return [
            {
                "payload": hit.get("payload", {}),
                "score": hit.get("score", 0.0),
                "match_type": "semantic"
            }
            for hit in search_results.get("result", [])
        ]


def _merge_results(
    file_name_results: List[Dict[str, Any]],
    semantic_results: List[Dict[str, Any]],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Merge file name and semantic results, removing duplicates.
    
    Priority: file name matches first, then semantic matches.
    """
    merged = {}
    
    # Add file name matches first (they get priority)
    for result in file_name_results:
        file_path = result["payload"].get("rel_path", "")
        if file_path not in merged:
            merged[file_path] = result
    
    # Add semantic matches (skip if already in file name results)
    for result in semantic_results:
        file_path = result["payload"].get("rel_path", "")
        if file_path not in merged:
            merged[file_path] = result
    
    # Convert back to list and sort by score
    final_results = list(merged.values())
    final_results.sort(key=lambda x: x["score"], reverse=True)
    
    return final_results[:top_k]


async def _get_local_embedding(text: str) -> Dict[str, Any]:
    """Generate embedding using local model."""
    import httpx
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{EMBED_SERVER_URL}/embed",
            json={"input": text, "type": "query"}
        )
        
        if response.status_code != 200:
            raise Exception(f"Embedding failed: {response.status_code}")
        
        result = response.json()
        return {
            "embedding": result["embedding"],
            "model": "Xenova/bge-small-en-v1.5",
            "dimension": len(result["embedding"])
        }


async def _check_embedding_server_health() -> bool:
    """Check if embedding server is ready."""
    import httpx
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{EMBED_SERVER_URL}/health")
            return response.status_code == 200 and response.json().get("ready", False)
    except:
        return False




@tool
async def code_snippets(
    query: str,
    config: RunnableConfig,
    filter_by: Optional[Dict[str, Any]] = None,
    top_k: int = 3,  # REDUCED from 5 to 3
    include_context: bool = True,
    include_full_code: bool = False,  # NEW: Default to signatures only
    max_code_chars: int = 1500,  # NEW: Limit code length when full code requested
    score_threshold: float = 0.30
) -> Dict[str, Any]:
    """Search C# scripts with token-efficient output.
    
    By default, returns metadata and code signatures to minimize tokens.
    Set include_full_code=True only when user explicitly asks for full code.
    
    Args:
        query: Search query (file name or description)
        top_k: Number of results (default 3, reduced for token efficiency)
        include_full_code: Return full code? (default False - saves tokens!)
        max_code_chars: Max code length when full code included (default 1500)
        score_threshold: Minimum semantic score (default 0.30)
        
    Returns:
        Snippets with metadata + signatures (or full code if requested)
    """
    try:
        # Get project context from config
        configurable = config.get("configurable", {})
        project_id = configurable.get("project_id")
        
        if not project_id:
            return {
                "success": False,
                "error": "Project ID not available",
                "query": query
            }
        
        # Check embedding server
        server_ready = await _check_embedding_server_health()
        if not server_ready:
            return {
                "success": False,
                "error": f"Embedding server not ready at {EMBED_SERVER_URL}",
                "query": query
            }
        
        # Two-stage search
        file_patterns = _extract_file_name_patterns(query)
        
        file_name_results = []
        if file_patterns:
            file_name_results = await _search_by_file_name(project_id, file_patterns, limit=top_k)
        
        semantic_results = await _search_by_semantic(
            project_id, query, limit=top_k, score_threshold=score_threshold
        )
        
        merged_results = _merge_results(file_name_results, semantic_results, top_k=top_k)
        
        # Create token-efficient snippets
        snippets = [
            _create_token_efficient_snippet(
                r["payload"],
                r["score"],
                r["match_type"],
                include_full_code=include_full_code,
                max_code_chars=max_code_chars
            )
            for r in merged_results
        ]
        
        # Calculate token usage estimate
        total_code_chars = sum(
            len(s.get("code", s.get("code_signature", ""))) 
            for s in snippets
        )
        estimated_tokens = total_code_chars // 4  # Rough estimate: 4 chars per token
        
        return {
            "success": True,
            "query": query,
            "file_patterns_extracted": file_patterns,
            "search_strategy": "two_stage_hybrid_token_efficient",
            "snippets": snippets,
            "total_found": len(snippets),
            "score_threshold": score_threshold,
            "project_id": project_id,
            "timestamp": datetime.now(UTC).isoformat(),
            
            # Token usage info
            "include_full_code": include_full_code,
            "estimated_tokens": estimated_tokens,
            "token_savings_mode": not include_full_code
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Code search failed: {str(e)}",
            "query": query,
            "error_type": type(e).__name__
        }