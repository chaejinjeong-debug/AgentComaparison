"""Built-in tools for the Agent Engine platform.

This module provides commonly used tools that can be registered with the agent.

Available tools:
- search: Web search functionality (mock implementation)
- calculate: Mathematical calculations
- get_current_datetime: Current date and time
- convert_timezone: Timezone conversion
- save_user_memory: Save user facts to memory (Phase 2)
- recall_user_info: Recall user information from memory (Phase 2)
- forget_user_info: Delete user memories - GDPR (Phase 2)
"""

from agent_engine.tools.calculator import calculate
from agent_engine.tools.datetime_tool import convert_timezone, get_current_datetime
from agent_engine.tools.memory_tools import (
    MEMORY_TOOLS,
    forget_user_info,
    recall_user_info,
    save_user_memory,
    set_current_user,
    set_memory_manager,
)
from agent_engine.tools.search import search

__all__ = [
    # Core tools
    "search",
    "calculate",
    "get_current_datetime",
    "convert_timezone",
    # Memory tools (Phase 2)
    "save_user_memory",
    "recall_user_info",
    "forget_user_info",
    "set_memory_manager",
    "set_current_user",
    "MEMORY_TOOLS",
]

# Default tools that can be easily registered with an agent
DEFAULT_TOOLS = [
    search,
    calculate,
    get_current_datetime,
    convert_timezone,
]

# All tools including Phase 2 memory tools
ALL_TOOLS = DEFAULT_TOOLS + MEMORY_TOOLS
