"""Built-in tools for the Agent Engine platform.

This module provides commonly used tools that can be registered with the agent.

Available tools:
- search: Web search functionality (mock implementation)
- calculate: Mathematical calculations
- get_current_datetime: Current date and time
- convert_timezone: Timezone conversion
"""

from agent_engine.tools.calculator import calculate
from agent_engine.tools.datetime_tool import convert_timezone, get_current_datetime
from agent_engine.tools.search import search

__all__ = [
    "search",
    "calculate",
    "get_current_datetime",
    "convert_timezone",
]

# Default tools that can be easily registered with an agent
DEFAULT_TOOLS = [
    search,
    calculate,
    get_current_datetime,
    convert_timezone,
]
