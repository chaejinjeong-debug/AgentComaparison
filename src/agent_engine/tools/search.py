"""Search tool implementation.

This module provides a mock search functionality for testing purposes.
In Phase 2+, this can be replaced with actual search API integration.
"""

from typing import Any

import structlog

logger = structlog.get_logger()

# Mock search results for demonstration
MOCK_SEARCH_RESULTS: dict[str, list[dict[str, str]]] = {
    "weather": [
        {
            "title": "Current Weather Conditions",
            "snippet": "Today's weather is sunny with a high of 25°C.",
            "url": "https://example.com/weather",
        },
        {
            "title": "Weather Forecast",
            "snippet": "The week ahead looks mostly sunny with temperatures between 20-28°C.",
            "url": "https://example.com/forecast",
        },
    ],
    "news": [
        {
            "title": "Latest Headlines",
            "snippet": "Stay updated with the latest news from around the world.",
            "url": "https://example.com/news",
        },
    ],
    "python": [
        {
            "title": "Python Documentation",
            "snippet": "Python is a programming language that lets you work quickly.",
            "url": "https://docs.python.org",
        },
        {
            "title": "Python Tutorial",
            "snippet": "Learn Python programming step by step with examples.",
            "url": "https://example.com/python-tutorial",
        },
    ],
}

DEFAULT_RESULT = [
    {
        "title": "Search Results",
        "snippet": "No specific results found for your query.",
        "url": "https://example.com/search",
    }
]


def search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search the web for information (mock implementation).

    This is a mock implementation that returns predefined results for testing.
    In production, this would integrate with a real search API like:
    - Google Custom Search API
    - Vertex AI Search
    - Bing Search API

    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 5)

    Returns:
        Dictionary containing:
            - query: The original search query
            - results: List of search result dictionaries with title, snippet, and url
            - count: Number of results returned
    """
    logger.info("search_executed", query=query, max_results=max_results)

    # Find matching mock results
    query_lower = query.lower()
    results = []

    for keyword, keyword_results in MOCK_SEARCH_RESULTS.items():
        if keyword in query_lower:
            results.extend(keyword_results)

    # Use default results if no matches
    if not results:
        results = DEFAULT_RESULT

    # Limit results
    results = results[:max_results]

    return {
        "query": query,
        "results": results,
        "count": len(results),
    }
