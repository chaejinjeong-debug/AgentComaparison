"""Memory tools for the Agent Engine platform.

These tools allow the agent to explicitly save and recall user information.
"""

from typing import Any

import structlog

from agent_engine.memory import MemoryManager, MemoryScope
from agent_engine.observability.decorators import logged_tool

logger = structlog.get_logger(__name__)

# Global memory manager instance (set during agent initialization)
_memory_manager: MemoryManager | None = None
_current_user_id: str | None = None


def set_memory_manager(manager: MemoryManager) -> None:
    """Set the global memory manager instance.

    Args:
        manager: MemoryManager instance
    """
    global _memory_manager
    _memory_manager = manager
    logger.debug("Memory manager set for tools")


def set_current_user(user_id: str) -> None:
    """Set the current user ID for memory operations.

    Args:
        user_id: User identifier
    """
    global _current_user_id
    _current_user_id = user_id
    logger.debug("Current user set for memory tools", user_id=user_id)


@logged_tool
def save_user_memory(
    fact: str,
    topics: list[str] | None = None,
) -> dict[str, Any]:
    """Save a fact about the user to memory.

    Use this tool when the user shares personal information that should
    be remembered for future conversations. Examples:
    - User preferences ("I prefer dark mode")
    - Personal details ("My name is John")
    - Work information ("I work at Acme Corp")

    Args:
        fact: The fact to remember about the user
        topics: Optional list of topics/tags for categorization

    Returns:
        Dictionary with status and memory_id
    """
    import asyncio

    if _memory_manager is None:
        logger.warning("Memory manager not initialized")
        return {
            "status": "error",
            "message": "Memory system not available",
        }

    if _current_user_id is None:
        logger.warning("Current user not set")
        return {
            "status": "error",
            "message": "User context not available",
        }

    try:
        # Run async operation in sync context
        loop = asyncio.get_event_loop()
        memory = loop.run_until_complete(
            _memory_manager.save_memory(
                user_id=_current_user_id,
                fact=fact,
                topics=topics or [],
                scope=MemoryScope.USER,
                source="agent",
            )
        )

        logger.info(
            "Memory saved by agent",
            memory_id=memory.memory_id,
            user_id=_current_user_id,
            fact_preview=fact[:50],
        )

        return {
            "status": "success",
            "message": f"I'll remember that: {fact}",
            "memory_id": memory.memory_id,
        }

    except Exception as e:
        logger.error("Failed to save memory", error=str(e))
        return {
            "status": "error",
            "message": f"Failed to save memory: {str(e)}",
        }


@logged_tool
def recall_user_info(
    query: str,
    max_results: int = 5,
) -> dict[str, Any]:
    """Recall information about the user from memory.

    Use this tool when you need to remember something about the user
    that was mentioned in previous conversations. Examples:
    - "What is the user's name?"
    - "What does the user work on?"
    - "What are the user's preferences?"

    Args:
        query: What to search for in user memories
        max_results: Maximum number of memories to return

    Returns:
        Dictionary with status and list of relevant memories
    """
    import asyncio

    if _memory_manager is None:
        logger.warning("Memory manager not initialized")
        return {
            "status": "error",
            "message": "Memory system not available",
            "memories": [],
        }

    if _current_user_id is None:
        logger.warning("Current user not set")
        return {
            "status": "error",
            "message": "User context not available",
            "memories": [],
        }

    try:
        # Run async operation in sync context
        loop = asyncio.get_event_loop()
        memories = loop.run_until_complete(
            _memory_manager.retrieve_memories(
                user_id=_current_user_id,
                query=query,
                max_results=max_results,
            )
        )

        if not memories:
            return {
                "status": "success",
                "message": "No relevant memories found",
                "memories": [],
            }

        memory_list = [
            {
                "fact": m.fact,
                "topics": m.topics,
                "created_at": m.created_at.isoformat(),
            }
            for m in memories
        ]

        logger.debug(
            "Memories recalled",
            user_id=_current_user_id,
            query=query,
            count=len(memories),
        )

        return {
            "status": "success",
            "message": f"Found {len(memories)} relevant memories",
            "memories": memory_list,
        }

    except Exception as e:
        logger.error("Failed to recall memories", error=str(e))
        return {
            "status": "error",
            "message": f"Failed to recall memories: {str(e)}",
            "memories": [],
        }


@logged_tool
def forget_user_info(
    memory_id: str | None = None,
    forget_all: bool = False,
) -> dict[str, Any]:
    """Delete user memories (GDPR compliance support).

    Use this tool when the user requests to delete their stored information.

    Args:
        memory_id: Specific memory ID to delete
        forget_all: If True, delete all user memories

    Returns:
        Dictionary with status and deletion count
    """
    import asyncio

    if _memory_manager is None:
        logger.warning("Memory manager not initialized")
        return {
            "status": "error",
            "message": "Memory system not available",
        }

    if _current_user_id is None:
        logger.warning("Current user not set")
        return {
            "status": "error",
            "message": "User context not available",
        }

    try:
        loop = asyncio.get_event_loop()

        if forget_all:
            count = loop.run_until_complete(_memory_manager.delete_user_memories(_current_user_id))
            logger.info(
                "All user memories deleted",
                user_id=_current_user_id,
                count=count,
            )
            return {
                "status": "success",
                "message": f"Deleted {count} memories",
                "count": count,
            }

        elif memory_id:
            success = loop.run_until_complete(_memory_manager.delete_memory(memory_id))
            if success:
                logger.info(
                    "Memory deleted",
                    memory_id=memory_id,
                    user_id=_current_user_id,
                )
                return {
                    "status": "success",
                    "message": "Memory deleted",
                    "count": 1,
                }
            else:
                return {
                    "status": "error",
                    "message": "Memory not found",
                    "count": 0,
                }

        else:
            return {
                "status": "error",
                "message": "Specify memory_id or set forget_all=True",
            }

    except Exception as e:
        logger.error("Failed to delete memories", error=str(e))
        return {
            "status": "error",
            "message": f"Failed to delete memories: {str(e)}",
        }


# Tool functions for pydantic-ai registration
MEMORY_TOOLS = [
    save_user_memory,
    recall_user_info,
    forget_user_info,
]
