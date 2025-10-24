"""Enhanced web search tool with SearXNG primary and Brave Search fallback.

This module provides a robust search implementation with:
- SearXNG as primary search engine (privacy-respecting, multi-engine aggregation)
- Brave Search API as fallback (reliable, fast results)
- Intelligent retry logic with exponential backoff
- Response caching to minimize API calls
- Structured error handling and logging
- Async/await support for non-blocking operations
- Rate limiting and timeout management
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Literal
import os
import asyncio
import time
from datetime import datetime, UTC, timedelta
from enum import Enum
from dataclasses import dataclass, field
import hashlib
import json

import aiohttp
import requests
from langchain_core.tools import tool

# Optional caching support
try:
    from cachetools import TTLCache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    print("Warning: cachetools not installed. Caching disabled. Install with: pip install cachetools")


class SearchEngine(str, Enum):
    """Available search engines."""
    SEARXNG = "searxng"
    BRAVE = "brave"


@dataclass
class SearchConfig:
    """Configuration for the enhanced search tool."""
    
    # SearXNG Configuration
    searxng_url: str = field(default_factory=lambda: os.getenv("SEARXNG_URL", "http://localhost:8888"))
    searxng_timeout: int = field(default_factory=lambda: int(os.getenv("SEARXNG_TIMEOUT", "15")))
    searxng_engines: Optional[str] = field(default_factory=lambda: os.getenv("SEARXNG_ENGINES"))  # e.g., "google,duckduckgo,brave"
    searxng_categories: Optional[str] = field(default_factory=lambda: os.getenv("SEARXNG_CATEGORIES"))  # e.g., "general,images"
    searxng_language: str = field(default_factory=lambda: os.getenv("SEARXNG_LANGUAGE", "en"))
    
    # Brave Search Configuration
    brave_api_key: Optional[str] = field(default_factory=lambda: os.getenv("BRAVE_SEARCH_API_KEY"))
    brave_timeout: int = field(default_factory=lambda: int(os.getenv("BRAVE_TIMEOUT", "10")))
    brave_base_url: str = "https://api.search.brave.com/res/v1/web/search"
    
    # Retry Configuration
    max_retries: int = 3
    retry_delay: float = 1.0  # Initial delay in seconds
    retry_exponential_base: float = 2.0  # Exponential backoff multiplier
    
    # Result Configuration
    max_results: int = field(default_factory=lambda: int(os.getenv("MAX_SEARCH_RESULTS", "10")))
    
    # Cache Configuration
    enable_cache: bool = field(default_factory=lambda: os.getenv("ENABLE_SEARCH_CACHE", "true").lower() == "true")
    cache_ttl: int = field(default_factory=lambda: int(os.getenv("SEARCH_CACHE_TTL", "3600")))  # 1 hour default
    cache_max_size: int = field(default_factory=lambda: int(os.getenv("SEARCH_CACHE_SIZE", "100")))
    
    # Fallback Configuration
    enable_brave_fallback: bool = True
    fallback_on_empty_results: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        # Validate SearXNG URL
        if not self.searxng_url:
            raise ValueError("SEARXNG_URL must be configured")
        
        # Validate Brave API key if fallback is enabled
        if self.enable_brave_fallback and not self.brave_api_key:
            print("Warning: BRAVE_SEARCH_API_KEY not set. Brave fallback will be disabled.")
            self.enable_brave_fallback = False


class SearchCache:
    """Simple TTL cache for search results."""
    
    def __init__(self, max_size: int = 100, ttl: int = 3600):
        """Initialize cache.
        
        Args:
            max_size: Maximum number of cached items
            ttl: Time to live in seconds
        """
        if CACHE_AVAILABLE:
            self._cache = TTLCache(maxsize=max_size, ttl=ttl)
        else:
            self._cache = {}
            self._timestamps = {}
            self._max_size = max_size
            self._ttl = ttl
    
    def _get_key(self, query: str, engine: SearchEngine, **kwargs) -> str:
        """Generate cache key from query parameters."""
        key_data = {
            "query": query,
            "engine": engine.value,
            **kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(self, query: str, engine: SearchEngine, **kwargs) -> Optional[Dict[str, Any]]:
        """Get cached result if available and not expired."""
        key = self._get_key(query, engine, **kwargs)
        
        if CACHE_AVAILABLE:
            return self._cache.get(key)
        else:
            # Manual TTL implementation
            if key in self._cache:
                timestamp = self._timestamps.get(key, 0)
                if time.time() - timestamp < self._ttl:
                    return self._cache[key]
                else:
                    # Expired, remove it
                    del self._cache[key]
                    del self._timestamps[key]
            return None
    
    def set(self, query: str, engine: SearchEngine, result: Dict[str, Any], **kwargs):
        """Cache a search result."""
        key = self._get_key(query, engine, **kwargs)
        
        if CACHE_AVAILABLE:
            self._cache[key] = result
        else:
            # Manual size and TTL management
            if len(self._cache) >= self._max_size:
                # Remove oldest entry
                oldest_key = min(self._timestamps, key=self._timestamps.get)
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]
            
            self._cache[key] = result
            self._timestamps[key] = time.time()
    
    def clear(self):
        """Clear all cached results."""
        if CACHE_AVAILABLE:
            self._cache.clear()
        else:
            self._cache.clear()
            self._timestamps.clear()


class EnhancedWebSearch:
    """Enhanced web search with SearXNG primary and Brave fallback."""
    
    def __init__(self, config: Optional[SearchConfig] = None):
        """Initialize the enhanced search tool.
        
        Args:
            config: Search configuration. If None, uses default configuration.
        """
        self.config = config or SearchConfig()
        self.cache = SearchCache(
            max_size=self.config.cache_max_size,
            ttl=self.config.cache_ttl
        ) if self.config.enable_cache else None
        
        # Statistics
        self.stats = {
            "total_searches": 0,
            "searxng_success": 0,
            "searxng_failures": 0,
            "brave_fallback_used": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    async def _retry_with_backoff(
        self,
        func,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with exponential backoff retry logic.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Result from func
            
        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(self.config.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (
                        self.config.retry_exponential_base ** attempt
                    )
                    print(f"Retry attempt {attempt + 1} after {delay:.2f}s: {str(e)}")
                    await asyncio.sleep(delay)
        
        raise last_exception
    
    async def _search_searxng(
        self,
        query: str,
        max_results: Optional[int] = None,
        time_range: Optional[Literal["day", "month", "year"]] = None,
        safe_search: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        """Search using SearXNG instance.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            time_range: Time range filter (day, month, year)
            safe_search: Safe search level (0=off, 1=moderate, 2=strict)
            **kwargs: Additional search parameters
            
        Returns:
            Structured search results
            
        Raises:
            aiohttp.ClientError: On network errors
            ValueError: On invalid response
        """
        params = {
            "q": query,
            "format": "json",
            "language": self.config.searxng_language,
            "safesearch": safe_search,
        }
        
        # Add optional parameters
        if self.config.searxng_engines:
            params["engines"] = self.config.searxng_engines
        
        if self.config.searxng_categories:
            params["categories"] = self.config.searxng_categories
        
        if time_range:
            params["time_range"] = time_range
        
        # Add any additional parameters
        params.update(kwargs)
        
        # Determine endpoint (prefer /search)
        url = f"{self.config.searxng_url.rstrip('/')}/search"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=self.config.searxng_timeout)
            ) as response:
                response.raise_for_status()
                data = await response.json()
        
        # Extract and normalize results
        results = data.get("results", [])
        
        # Limit results if specified
        if max_results:
            results = results[:max_results]
        elif self.config.max_results:
            results = results[:self.config.max_results]
        
        return {
            "success": True,
            "engine": SearchEngine.SEARXNG.value,
            "query": query,
            "results": [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                    "engine": r.get("engine", "unknown"),
                    "score": r.get("score", 0),
                    "category": r.get("category", "general"),
                    "publishedDate": r.get("publishedDate")
                }
                for r in results
            ],
            "result_count": len(results),
            "timestamp": datetime.now(UTC).isoformat(),
            "infoboxes": data.get("infoboxes", []),
            "suggestions": data.get("suggestions", []),
            "answers": data.get("answers", []),
            "corrections": data.get("corrections", [])
        }
    
    async def _search_brave(
        self,
        query: str,
        max_results: Optional[int] = None,
        safe_search: str = "moderate",
        **kwargs
    ) -> Dict[str, Any]:
        """Search using Brave Search API.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            safe_search: Safe search level (off, moderate, strict)
            **kwargs: Additional search parameters
            
        Returns:
            Structured search results
            
        Raises:
            aiohttp.ClientError: On network errors
            ValueError: On invalid response or missing API key
        """
        if not self.config.brave_api_key:
            raise ValueError("Brave API key not configured")
        
        params = {
            "q": query,
            "count": max_results or self.config.max_results,
            "safesearch": safe_search,
            "extra_snippets": True,
        }
        
        # Add any additional parameters
        params.update(kwargs)
        
        headers = {
            "X-Subscription-Token": self.config.brave_api_key,
            "Accept": "application/json",
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.config.brave_base_url,
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.config.brave_timeout)
            ) as response:
                response.raise_for_status()
                data = await response.json()
        
        # Extract and normalize results
        web_results = data.get("web", {}).get("results", [])
        
        return {
            "success": True,
            "engine": SearchEngine.BRAVE.value,
            "query": query,
            "results": [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": " ".join(filter(None, [
                        r.get("description", ""),
                        *r.get("extra_snippets", [])
                    ])),
                    "engine": "brave",
                    "score": 0,  # Brave doesn't provide scores
                    "category": "general",
                    "publishedDate": r.get("age")
                }
                for r in web_results
            ],
            "result_count": len(web_results),
            "timestamp": datetime.now(UTC).isoformat(),
            "infoboxes": [],
            "suggestions": [],
            "answers": [],
            "corrections": []
        }
    
    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        time_range: Optional[Literal["day", "month", "year"]] = None,
        safe_search: int = 0,
        force_engine: Optional[SearchEngine] = None,
        use_cache: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute search with automatic fallback.
        
        This method attempts to search using SearXNG first, and falls back
        to Brave Search if SearXNG fails or returns no results.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            time_range: Time range filter (day, month, year) - SearXNG only
            safe_search: Safe search level (0=off, 1=moderate, 2=strict)
            force_engine: Force specific search engine (skip fallback)
            use_cache: Whether to use cached results if available
            **kwargs: Additional search parameters
            
        Returns:
            Structured search results with metadata
            
        Example:
            >>> search = EnhancedWebSearch()
            >>> results = await search.search("Unity game development best practices")
            >>> print(f"Found {results['result_count']} results using {results['engine']}")
        """
        self.stats["total_searches"] += 1
        
        # Check cache first
        if use_cache and self.cache:
            cache_key_params = {
                "max_results": max_results,
                "time_range": time_range,
                "safe_search": safe_search,
                **kwargs
            }
            
            # Try SearXNG cache first
            cached_result = self.cache.get(
                query,
                SearchEngine.SEARXNG,
                **cache_key_params
            )
            
            if cached_result:
                self.stats["cache_hits"] += 1
                cached_result["from_cache"] = True
                return cached_result
            
            # Try Brave cache if fallback enabled
            if self.config.enable_brave_fallback:
                cached_result = self.cache.get(
                    query,
                    SearchEngine.BRAVE,
                    **cache_key_params
                )
                
                if cached_result:
                    self.stats["cache_hits"] += 1
                    cached_result["from_cache"] = True
                    return cached_result
            
            self.stats["cache_misses"] += 1
        
        # If force_engine is specified, only use that engine
        if force_engine == SearchEngine.BRAVE:
            return await self._execute_brave_search(
                query, max_results, safe_search, use_cache, **kwargs
            )
        
        # Try SearXNG first
        try:
            result = await self._retry_with_backoff(
                self._search_searxng,
                query,
                max_results=max_results,
                time_range=time_range,
                safe_search=safe_search,
                **kwargs
            )
            
            self.stats["searxng_success"] += 1
            
            # Cache successful result
            if use_cache and self.cache:
                self.cache.set(
                    query,
                    SearchEngine.SEARXNG,
                    result,
                    max_results=max_results,
                    time_range=time_range,
                    safe_search=safe_search,
                    **kwargs
                )
            
            # Check if we got results
            if result["result_count"] > 0:
                result["from_cache"] = False
                return result
            
            # If no results and fallback on empty is enabled, try Brave
            if self.config.fallback_on_empty_results and self.config.enable_brave_fallback:
                print(f"SearXNG returned no results for '{query}', trying Brave fallback...")
                return await self._execute_brave_search(
                    query, max_results, safe_search, use_cache, **kwargs
                )
            
            result["from_cache"] = False
            return result
            
        except Exception as searxng_error:
            self.stats["searxng_failures"] += 1
            print(f"SearXNG search failed: {str(searxng_error)}")
            
            # Try Brave fallback if enabled
            if self.config.enable_brave_fallback and force_engine != SearchEngine.SEARXNG:
                print(f"Attempting Brave Search fallback...")
                try:
                    return await self._execute_brave_search(
                        query, max_results, safe_search, use_cache, **kwargs
                    )
                except Exception as brave_error:
                    # Both failed, return error
                    return {
                        "success": False,
                        "error": f"All search engines failed. SearXNG: {str(searxng_error)}, Brave: {str(brave_error)}",
                        "query": query,
                        "engines_tried": [SearchEngine.SEARXNG.value, SearchEngine.BRAVE.value],
                        "timestamp": datetime.now(UTC).isoformat()
                    }
            
            # No fallback available or forced engine
            return {
                "success": False,
                "error": f"SearXNG search failed: {str(searxng_error)}",
                "query": query,
                "engines_tried": [SearchEngine.SEARXNG.value],
                "timestamp": datetime.now(UTC).isoformat()
            }
    
    async def _execute_brave_search(
        self,
        query: str,
        max_results: Optional[int],
        safe_search: int,
        use_cache: bool,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute Brave search with retry and caching."""
        self.stats["brave_fallback_used"] += 1
        
        # Convert safe_search int to Brave format
        brave_safe_search = ["off", "moderate", "strict"][min(safe_search, 2)]
        
        result = await self._retry_with_backoff(
            self._search_brave,
            query,
            max_results=max_results,
            safe_search=brave_safe_search,
            **kwargs
        )
        
        # Cache successful result
        if use_cache and self.cache:
            self.cache.set(
                query,
                SearchEngine.BRAVE,
                result,
                max_results=max_results,
                safe_search=safe_search,
                **kwargs
            )
        
        result["from_cache"] = False
        result["fallback_used"] = True
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search statistics.
        
        Returns:
            Dictionary with search statistics
        """
        return {
            **self.stats,
            "cache_hit_rate": (
                self.stats["cache_hits"] / 
                (self.stats["cache_hits"] + self.stats["cache_misses"])
                if (self.stats["cache_hits"] + self.stats["cache_misses"]) > 0
                else 0.0
            ),
            "searxng_success_rate": (
                self.stats["searxng_success"] /
                (self.stats["searxng_success"] + self.stats["searxng_failures"])
                if (self.stats["searxng_success"] + self.stats["searxng_failures"]) > 0
                else 0.0
            ),
            "brave_fallback_rate": (
                self.stats["brave_fallback_used"] / self.stats["total_searches"]
                if self.stats["total_searches"] > 0
                else 0.0
            )
        }
    
    def clear_cache(self):
        """Clear the search result cache."""
        if self.cache:
            self.cache.clear()


# Global search instance (lazy initialization)
_search_instance: Optional[EnhancedWebSearch] = None


def get_search_instance() -> EnhancedWebSearch:
    """Get or create the global search instance.
    
    Returns:
        Configured EnhancedWebSearch instance
    """
    global _search_instance
    if _search_instance is None:
        _search_instance = EnhancedWebSearch()
    return _search_instance


@tool
async def web_search(
    query: str,
    max_results: int = 10,
    time_range: Optional[str] = None,
    safe_search: int = 0
) -> Dict[str, Any]:
    """Search the web for information about game development, Unity, and related topics.
    
    Uses SearXNG as the primary search engine with automatic fallback to Brave Search
    if needed. Results are cached to improve performance.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 10)
        time_range: Optional time range filter - "day", "month", or "year"
        safe_search: Safe search level - 0 (off), 1 (moderate), or 2 (strict)
        
    Returns:
        Search results with URLs, titles, and content. Includes metadata about
        the search engine used, number of results, and any suggestions or answers.
        
    Example:
        >>> result = await web_search("Unity coroutines best practices")
        >>> for item in result["results"]:
        >>>     print(f"{item['title']}: {item['url']}")
    """
    try:
        search = get_search_instance()
        
        # Validate time_range
        valid_time_ranges = ["day", "month", "year"]
        if time_range and time_range not in valid_time_ranges:
            return {
                "success": False,
                "error": f"Invalid time_range. Must be one of: {', '.join(valid_time_ranges)}",
                "query": query
            }
        
        return await search.search(
            query=query,
            max_results=max_results,
            time_range=time_range,
            safe_search=safe_search
        )
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Web search failed: {str(e)}",
            "query": query,
            "timestamp": datetime.now(UTC).isoformat()
        }


# Synchronous wrapper for backward compatibility
def web_search_sync(
    query: str,
    max_results: int = 10,
    time_range: Optional[str] = None,
    safe_search: int = 0
) -> Dict[str, Any]:
    """Synchronous wrapper for web_search.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        time_range: Optional time range filter
        safe_search: Safe search level
        
    Returns:
        Search results dictionary
    """
    return asyncio.run(web_search(query, max_results, time_range, safe_search))


if __name__ == "__main__":
    # Example usage and testing
    async def test_search():
        """Test the enhanced search functionality."""
        print("Enhanced Web Search Test\n" + "="*50)
        
        search = EnhancedWebSearch()
        
        # Test query
        query = "Unity game development best practices"
        print(f"\nSearching for: {query}")
        print("-" * 50)
        
        result = await search.search(query, max_results=5)
        
        if result["success"]:
            print(f"\n✓ Search successful using {result['engine']}")
            print(f"Found {result['result_count']} results")
            
            if result.get("from_cache"):
                print("(Results from cache)")
            
            if result.get("fallback_used"):
                print("(Used fallback engine)")
            
            print("\nResults:")
            for i, item in enumerate(result["results"][:3], 1):
                print(f"\n{i}. {item['title']}")
                print(f"   URL: {item['url']}")
                print(f"   Engine: {item['engine']}")
                print(f"   Preview: {item['content'][:150]}...")
            
            if result.get("suggestions"):
                print(f"\nSuggestions: {', '.join(result['suggestions'])}")
            
            if result.get("answers"):
                print(f"\nDirect answers found: {len(result['answers'])}")
        else:
            print(f"\n✗ Search failed: {result.get('error')}")
        
        # Print statistics
        print("\n" + "="*50)
        print("Statistics:")
        stats = search.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    # Run test
    asyncio.run(test_search())