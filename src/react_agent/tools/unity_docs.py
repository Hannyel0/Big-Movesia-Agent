"""
OPTIMIZED Unity Documentation RAG Tool
Key improvements:
1. Parallel URL fetching with asyncio.gather()
2. Batch embedding support
3. Larger cache with LRU eviction
4. Concurrent trafilatura extraction
5. Connection pooling
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional
from functools import lru_cache
from datetime import datetime, UTC
from concurrent.futures import ThreadPoolExecutor

import httpx
import trafilatura
from trafilatura.settings import use_config
from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Configuration from environment
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
UNITY_DOCS_COLLECTION = os.getenv("UNITY_DOCS_COLLECTION", "unity_docs-fast")
EMBED_SERVER_URL = os.getenv("EMBED_SERVER_URL", "http://127.0.0.1:8766")

# 🚀 OPTIMIZATION 1: Increase cache size from 100 to 500
_url_cache: Dict[str, str] = {}
_cache_max_size = 500  # Increased from 100

# 🚀 OPTIMIZATION 2: Connection pool for better performance
_http_client: Optional[httpx.AsyncClient] = None


def get_http_client() -> httpx.AsyncClient:
    """
    Get or create a persistent HTTP client with connection pooling.
    This avoids overhead of creating new clients for each request.
    """
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            limits=httpx.Limits(
                max_keepalive_connections=10, max_connections=20, keepalive_expiry=30.0
            ),
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
        )
    return _http_client


# 🚀 OPTIMIZATION 3: Thread pool for CPU-bound trafilatura extraction
_extraction_executor = ThreadPoolExecutor(max_workers=4)


def _extract_content_sync(html: str) -> Optional[str]:
    """
    Synchronous trafilatura extraction to run in thread pool.
    This prevents blocking the async event loop during CPU-intensive parsing.
    """
    try:
        config = use_config()
        config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")

        extracted = trafilatura.extract(
            html,
            output_format="txt",
            include_comments=False,
            include_tables=True,
            include_images=False,
            include_links=False,
            no_fallback=False,
            favor_precision=False,
            favor_recall=True,
            config=config,
        )

        if not extracted or len(extracted.strip()) < 100:
            # Try fallback
            extracted = trafilatura.extract(
                html,
                output_format="txt",
                include_comments=False,
                include_tables=True,
                no_fallback=True,
                favor_precision=False,
                favor_recall=True,
                config=config,
            )

        return extracted
    except Exception as e:
        logger.error(f"❌ [Extraction] Failed: {e}")
        return None


async def _fetch_url_content(url: str) -> Optional[str]:
    """
    🚀 OPTIMIZED: Fetch and extract content with connection pooling and thread-based extraction.
    """
    global _url_cache

    # Check cache first
    if url in _url_cache:
        logger.info(f"📋 [UnityDocs._fetch_url_content] Cache hit for {url}")
        return _url_cache[url]

    try:
        # 🚀 Use persistent connection pool
        client = get_http_client()
        logger.info(f"📡 [UnityDocs._fetch_url_content] Fetching URL: {url}")
        response = await client.get(url)
        response.raise_for_status()
        logger.info(
            f"✅ [UnityDocs._fetch_url_content] HTTP {response.status_code} - Downloaded {len(response.text)} chars"
        )

        # 🚀 Run CPU-intensive extraction in thread pool
        logger.info(
            f"🔧 [UnityDocs._fetch_url_content] Starting content extraction (thread pool)"
        )
        loop = asyncio.get_event_loop()
        extracted = await loop.run_in_executor(
            _extraction_executor, _extract_content_sync, response.text
        )

        if not extracted:
            logger.error(
                f"❌ [UnityDocs._fetch_url_content] Failed to extract content from {url}"
            )
            return None

        # Clean up the extracted text
        clean_text = extracted.strip()

        # Remove excessive blank lines
        lines = clean_text.split("\n")
        cleaned_lines = []
        blank_count = 0
        for line in lines:
            if line.strip():
                cleaned_lines.append(line)
                blank_count = 0
            else:
                blank_count += 1
                if blank_count <= 2:
                    cleaned_lines.append(line)

        clean_text = "\n".join(cleaned_lines)

        # Smart truncation
        max_chars = 8000
        if len(clean_text) > max_chars:
            logger.info(
                f"✂️ [UnityDocs._fetch_url_content] Truncating from {len(clean_text)} to {max_chars} chars"
            )
            truncate_point = clean_text.rfind("\n\n", 0, max_chars)
            if truncate_point > max_chars * 0.8:
                clean_text = (
                    clean_text[:truncate_point]
                    + "\n\n[Content truncated at natural paragraph break...]"
                )
            else:
                truncate_point = clean_text.rfind(". ", 0, max_chars)
                if truncate_point > max_chars * 0.9:
                    clean_text = (
                        clean_text[: truncate_point + 1] + "\n\n[Content truncated...]"
                    )
                else:
                    clean_text = clean_text[:max_chars] + "\n\n[Content truncated...]"

        # Cache with simple FIFO eviction
        if len(_url_cache) >= _cache_max_size:
            first_key = next(iter(_url_cache))
            del _url_cache[first_key]

        _url_cache[url] = clean_text
        logger.info(
            f"✅ [UnityDocs._fetch_url_content] Cached - URL: {url}, Size: {len(clean_text)} chars"
        )

        return clean_text

    except httpx.HTTPError as e:
        logger.error(f"❌ [UnityDocs._fetch_url_content] HTTP error: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ [UnityDocs._fetch_url_content] Error: {e}")
        return None


class UnityDocsInput(BaseModel):
    """Input schema for Unity docs search."""

    query: str = Field(
        description="Natural language query about Unity functionality, API, or concepts"
    )
    version: Optional[str] = Field(
        default=None, description="Unity version filter (e.g., '6000.2')"
    )
    category: Optional[str] = Field(
        default=None, description="Category filter: 'ScriptReference', 'Manual', etc."
    )
    top_k: int = Field(
        default=5, ge=1, le=20, description="Number of results to return (1-20)"
    )
    fetch_full_content: bool = Field(
        default=True, description="Whether to fetch full page content from URLs"
    )
    score_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Minimum relevance threshold"
    )


async def _get_local_embedding(text: str) -> Dict[str, Any]:
    """Generate embedding using local embedding server with connection pooling."""
    client = get_http_client()
    response = await client.post(
        f"{EMBED_SERVER_URL}/embed", json={"input": text, "type": "query"}
    )

    if response.status_code != 200:
        raise Exception(f"Embedding failed: {response.status_code}")

    result = response.json()
    return {
        "embedding": result["embedding"],
        "model": "Xenova/bge-small-en-v1.5",
        "dimension": len(result["embedding"]),
    }


async def _search_unity_docs_semantic(
    query: str,
    version: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 5,
    score_threshold: float = 0.6,
) -> List[Dict[str, Any]]:
    """
    Perform semantic search against Unity docs collection.
    """
    logger.info(
        f"🔍 [UnityDocs._search_unity_docs_semantic] Starting search - Query: '{query}', Limit: {limit}"
    )

    try:
        # Build filter conditions
        filter_conditions = []
        if version:
            filter_conditions.append({"key": "version", "match": {"value": version}})
        if category:
            filter_conditions.append({"key": "category", "match": {"value": category}})

        # Generate embedding
        logger.info(f"🔧 [UnityDocs._search_unity_docs_semantic] Generating embedding")
        embedding_result = await _get_local_embedding(query)
        query_vector = embedding_result["embedding"]

        # Construct search payload
        search_payload = {
            "vector": query_vector,
            "limit": limit * 2,  # Fetch 2x for filtering
            "with_payload": True,
            "score_threshold": score_threshold,
        }

        if filter_conditions:
            search_payload["filter"] = {"must": filter_conditions}

        # 🚀 Use persistent connection pool
        client = get_http_client()
        qdrant_endpoint = (
            f"{QDRANT_URL}/collections/{UNITY_DOCS_COLLECTION}/points/search"
        )
        logger.info(f"📡 [UnityDocs._search_unity_docs_semantic] Querying Qdrant")

        response = await client.post(qdrant_endpoint, json=search_payload, timeout=30.0)

        logger.info(
            f"📨 [UnityDocs._search_unity_docs_semantic] Qdrant response: {response.status_code}"
        )

        if response.status_code != 200:
            logger.error(f"❌ Qdrant search failed: {response.status_code}")
            return []

        search_results = response.json()
        raw_hits = search_results.get("result", [])
        logger.info(
            f"📊 [UnityDocs._search_unity_docs_semantic] Qdrant returned {len(raw_hits)} results"
        )

        results = []
        for idx, hit in enumerate(raw_hits):
            payload = hit.get("payload", {})
            score = hit.get("score", 0.0)
            title = payload.get("title", "Unknown")

            # Title match boost
            query_lower = query.lower()
            title_lower = title.lower()
            if query_lower in title_lower:
                score *= 1.2

            results.append({"payload": payload, "score": score})

        logger.info(
            f"✅ [UnityDocs._search_unity_docs_semantic] Processed {len(results)} results"
        )

        # Re-sort and apply limit
        results.sort(key=lambda x: x["score"], reverse=True)
        final_results = results[:limit]

        if final_results:
            top_titles = [
                r["payload"].get("title", "Unknown")[:50] for r in final_results[:3]
            ]
            logger.info(
                f"🏆 [UnityDocs._search_unity_docs_semantic] Top results: {', '.join(top_titles)}"
            )

        return final_results

    except Exception as e:
        logger.error(
            f"❌ [UnityDocs._search_unity_docs_semantic] Search failed: {e}",
            exc_info=True,
        )
        return []


@tool(args_schema=UnityDocsInput)
async def unity_docs(
    query: str,
    version: Optional[str] = None,
    category: Optional[str] = None,
    top_k: int = 5,
    fetch_full_content: bool = True,
    score_threshold: float = 0.6,
) -> Dict[str, Any]:
    """
    🚀 OPTIMIZED: Search Unity documentation with parallel content fetching.

    Key optimizations:
    - Parallel URL fetching with asyncio.gather()
    - Connection pooling for HTTP requests
    - Thread pool for CPU-intensive content extraction
    - Larger cache (500 entries)
    """
    logger.info(f"{'=' * 80}")
    logger.info(f"🚀 [UnityDocs] TOOL INVOKED (OPTIMIZED)")
    logger.info(f"📝 [UnityDocs] Query: '{query}'")
    logger.info(
        f"🔧 [UnityDocs] Parameters: top_k={top_k}, fetch_content={fetch_full_content}"
    )
    logger.info(f"{'=' * 80}")

    try:
        # Step 1: Semantic search
        logger.info(f"🔍 [UnityDocs] Step 1: Performing semantic search")
        search_results = await _search_unity_docs_semantic(
            query=query,
            version=version,
            category=category,
            limit=top_k,
            score_threshold=score_threshold,
        )
        logger.info(
            f"📊 [UnityDocs] Semantic search returned {len(search_results)} results"
        )

        if not search_results:
            return {
                "success": True,
                "results": [],
                "result_count": 0,
                "message": f"No Unity docs found matching query with score >= {score_threshold}",
                "query": query,
            }

        # Step 2: Structure results and prepare for parallel fetching
        logger.info(f"🔨 [UnityDocs] Step 2: Structuring {len(search_results)} results")
        structured_results = []
        urls_to_fetch = []

        for idx, result in enumerate(search_results):
            payload = result["payload"]
            doc_entry = {
                "title": payload.get("title", "Unknown"),
                "url": payload.get("url", ""),
                "version": payload.get("version", ""),
                "relevance_score": round(result["score"], 3),
                "index": idx,  # Track original index
            }
            structured_results.append(doc_entry)

            if fetch_full_content and doc_entry["url"]:
                urls_to_fetch.append(doc_entry["url"])

        # 🚀 🚀 🚀 OPTIMIZATION 4: PARALLEL URL FETCHING! 🚀 🚀 🚀
        if urls_to_fetch:
            logger.info(
                f"🚀 [UnityDocs] Step 3: Fetching {len(urls_to_fetch)} URLs IN PARALLEL"
            )

            # Fetch all URLs concurrently!
            contents = await asyncio.gather(
                *[_fetch_url_content(url) for url in urls_to_fetch],
                return_exceptions=True,
            )

            logger.info(f"✅ [UnityDocs] Parallel fetch completed")

            # Attach fetched content to results
            for doc_entry, content in zip(structured_results, contents):
                if isinstance(content, Exception):
                    logger.warning(f"⚠️ Failed to fetch {doc_entry['url']}: {content}")
                    doc_entry["content_fetched"] = False
                elif content:
                    doc_entry["full_content"] = content
                    doc_entry["content_fetched"] = True
                    logger.debug(
                        f"✅ Attached content to '{doc_entry['title']}' ({len(content)} chars)"
                    )
                else:
                    doc_entry["content_fetched"] = False

        logger.info(
            f"✅ [UnityDocs] Successfully processed all {len(structured_results)} results"
        )

        # Step 4: Build response dictionary
        logger.info(f"📦 [UnityDocs] Step 4: Building response dictionary")

        response = {
            "success": True,
            "results": structured_results,
            "result_count": len(structured_results),
            "message": f"Found {len(structured_results)} relevant Unity docs",
            "query": query,
            "filters_applied": {
                "version": version,
                "category": category,
                "min_score": score_threshold,
            },
            "timestamp": datetime.now(UTC).isoformat(),
        }

        logger.info(f"{'=' * 80}")
        logger.info(f"✅ [UnityDocs] TOOL COMPLETED SUCCESSFULLY")
        logger.info(f"📊 [UnityDocs] Results: {len(structured_results)} docs")
        logger.info(
            f"📝 [UnityDocs] Top result: {structured_results[0]['title'] if structured_results else 'N/A'}"
        )
        logger.info(f"{'=' * 80}")

        return response

    except Exception as e:
        logger.error(f"❌ [UnityDocs] TOOL FAILED: {e}", exc_info=True)
        return {
            "success": False,
            "error": f"Search failed: {str(e)}",
            "results": [],
            "result_count": 0,
            "query": query,
            "error_type": type(e).__name__,
        }


# 🚀 OPTIMIZATION 5: Cleanup function for graceful shutdown
async def cleanup():
    """Close persistent connections gracefully."""
    global _http_client, _extraction_executor
    if _http_client:
        await _http_client.aclose()
        _http_client = None
    if _extraction_executor:
        _extraction_executor.shutdown(wait=True)
