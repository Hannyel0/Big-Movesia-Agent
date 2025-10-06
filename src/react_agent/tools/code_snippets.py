"""Code snippets tool using Qdrant vector search with local embeddings."""

from __future__ import annotations
from typing import Optional, Dict, Any, List
import os
from datetime import datetime, UTC

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import get_runtime

from react_agent.context import Context


# Local embedding server configuration
EMBED_SERVER_URL = os.getenv("EMBED_SERVER_URL", "http://127.0.0.1:8766")
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "movesia")


async def _get_local_embedding(text: str) -> Dict[str, Any]:
    """Generate embedding using local Xenova/bge-small-en-v1.5 model.
    
    Connects to the embed-server.ts running locally on port 8766.
    """
    import httpx
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Call local embedding server with 'query' type for search queries
            response = await client.post(
                f"{EMBED_SERVER_URL}/embed",
                json={
                    "input": text,
                    "type": "query"  # Important: use 'query' type for search
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Embedding server returned {response.status_code}: {response.text}")
            
            result = response.json()
            
            # Handle single embedding response format
            if "embedding" in result:
                return {
                    "embedding": result["embedding"],
                    "model": "Xenova/bge-small-en-v1.5",
                    "dimension": len(result["embedding"])
                }
            else:
                raise Exception("Unexpected embedding response format")
                
    except httpx.ConnectError:
        raise Exception(
            f"Cannot connect to local embedding server at {EMBED_SERVER_URL}. "
            "Make sure the embed server is running (npm run embed-server)"
        )
    except httpx.TimeoutException:
        raise Exception("Embedding request timed out. The model might still be loading.")
    except Exception as e:
        raise Exception(f"Embedding generation failed: {str(e)}")


async def _check_embedding_server_health() -> bool:
    """Check if the local embedding server is ready."""
    import httpx
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{EMBED_SERVER_URL}/health")
            if response.status_code == 200:
                health = response.json()
                return health.get("ready", False)
    except:
        pass
    return False


def _extract_code_context(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract meaningful context from code snippet payload."""
    code_text = payload.get("text", "")
    rel_path = payload.get("rel_path", "")
    
    # Extract file name and extension
    file_name = rel_path.split("/")[-1] if rel_path else "unknown"
    file_extension = file_name.split(".")[-1] if "." in file_name else ""
    
    # Extract basic code structure info
    context = {
        "file_name": file_name,
        "file_type": file_extension,
        "line_count": len(code_text.split("\n")),
        "has_class": "class " in code_text.lower(),
        "has_function": "void " in code_text or "public " in code_text or "private " in code_text,
        "has_unity_callbacks": any(cb in code_text for cb in ["Start()", "Update()", "Awake()", "FixedUpdate()"]),
    }
    
    # Try to extract class/namespace names
    lines = code_text.split("\n")
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.startswith("namespace "):
            context["namespace"] = line_stripped.replace("namespace ", "").replace("{", "").strip()
        if line_stripped.startswith("public class ") or line_stripped.startswith("class "):
            class_name = line_stripped.split("class ")[1].split(":")[0].split("{")[0].strip()
            context["class_name"] = class_name
    
    return context


@tool
async def code_snippets(
    query: str,
    config: RunnableConfig,
    filter_by: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
    include_context: bool = True,
    score_threshold: float = 0.20
) -> Dict[str, Any]:
    """Search through Unity C# scripts using semantic meaning via vector search.
    
    Finds code by WHAT IT DOES, not just what it's called. Uses local Xenova/bge-small-en-v1.5
    embeddings and Qdrant vector database to perform semantic search through indexed scripts.
    
    Args:
        query: Natural language description of what code you're looking for
               Examples: "player movement controller", "UI button click handler", 
                        "inventory system", "enemy AI pathfinding"
        filter_by: Optional filters (file_type, namespace, etc.)
        top_k: Number of results to return (default 5, max 20)
        include_context: Include surrounding code context and metadata
        score_threshold: Minimum similarity score (0.0-1.0, default 0.20)
        
    Returns:
        Ranked scripts with relevance scores and code snippets
    """
    try:
        # Get project context from config
        configurable = config.get("configurable", {})
        project_id = configurable.get("project_id")
        
        if not project_id:
            return {
                "success": False,
                "error": "Project ID not available. Ensure Unity project is connected.",
                "query": query
            }
        
        # Health check for embedding server
        server_ready = await _check_embedding_server_health()
        if not server_ready:
            return {
                "success": False,
                "error": (
                    f"Local embedding server not ready at {EMBED_SERVER_URL}. "
                    "Please start the embedding server: npm run embed-server"
                ),
                "query": query,
                "server_url": EMBED_SERVER_URL
            }
        
        # Generate query embedding using local model
        embedding_result = await _get_local_embedding(query)
        query_vector = embedding_result["embedding"]
        
        # Validate embedding dimension (should be 384 for bge-small-en-v1.5)
        expected_dim = 384
        if len(query_vector) != expected_dim:
            return {
                "success": False,
                "error": f"Embedding dimension mismatch: got {len(query_vector)}, expected {expected_dim}",
                "query": query
            }
        
        # Build Qdrant filter
        filter_conditions = [
            {"key": "project_id", "match": {"value": project_id}},
            {"key": "kind", "match": {"value": "Script"}}
        ]
        
        # Add additional filters if provided
        if filter_by:
            for key, value in filter_by.items():
                filter_conditions.append({
                    "key": key,
                    "match": {"value": value}
                })
        
        # Search Qdrant with local embeddings
        import httpx
        async with httpx.AsyncClient(timeout=30.0) as client:
            search_payload = {
                "vector": query_vector,
                "limit": min(top_k, 20),  # Cap at 20 results
                "with_payload": True,
                "score_threshold": score_threshold,
                "filter": {
                    "must": filter_conditions
                }
            }
            
            response = await client.post(
                f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/search",
                json=search_payload,
                timeout=30.0
            )
            
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Qdrant search failed: {response.status_code} - {response.text}",
                    "query": query
                }
            
            search_results = response.json()
        
        # Process and format results with rich context
        snippets = []
        for hit in search_results.get("result", []):
            payload = hit.get("payload", {})
            score = hit.get("score", 0.0)
            
            # Extract code snippet info
            snippet = {
                "file_path": payload.get("rel_path", "unknown"),
                "line_range": payload.get("range", ""),
                "relevance_score": round(score, 3),
                "code": payload.get("text", ""),
                "file_hash": payload.get("file_hash", ""),
            }
            
            # Add rich context if requested
            if include_context:
                code_context = _extract_code_context(payload)
                snippet.update({
                    "file_name": code_context.get("file_name", ""),
                    "file_type": code_context.get("file_type", ""),
                    "line_count": code_context.get("line_count", 0),
                    "has_class": code_context.get("has_class", False),
                    "has_unity_callbacks": code_context.get("has_unity_callbacks", False),
                    "class_name": code_context.get("class_name", ""),
                    "namespace": code_context.get("namespace", ""),
                })
            
            snippets.append(snippet)
        
        # Sort by relevance score (highest first)
        snippets.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return {
            "success": True,
            "query": query,
            "snippets": snippets,
            "total_found": len(snippets),
            "embedding_model": embedding_result.get("model", "unknown"),
            "score_threshold": score_threshold,
            "project_id": project_id,
            "timestamp": datetime.now(UTC).isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Code search failed: {str(e)}",
            "query": query,
            "error_type": type(e).__name__
        }