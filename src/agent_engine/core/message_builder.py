"""Message Builder for composing agent input messages.

This module provides utilities for building structured messages
that include context, memories, and session history.
"""

from __future__ import annotations

from typing import Any


class MessageBuilder:
    """Utility class for building agent input messages.

    This class handles the composition of user messages with additional
    context like session history, memories, and custom context data.

    The message format follows a structured template:
    - [Previous conversation] - Session history
    - [Relevant memories about the user] - Retrieved memories
    - [Additional context] - Custom context data
    - User message

    Example:
        >>> builder = MessageBuilder()
        >>> message = builder.build(
        ...     message="Hello",
        ...     memories=["User prefers formal language"],
        ...     context={"user_name": "John"},
        ... )
        >>> print(message)
        [Relevant memories about the user]
        - User prefers formal language

        [Additional context]
        - user_name: John

        Hello
    """

    # Section headers for message composition
    HISTORY_HEADER = "[Previous conversation]"
    MEMORIES_HEADER = "[Relevant memories about the user]"
    CONTEXT_HEADER = "[Additional context]"

    def build(
        self,
        message: str,
        context: dict[str, Any] | None = None,
        memories: list[str] | None = None,
        session_history: list[dict[str, str]] | None = None,
    ) -> str:
        """Build the full message with context, memories, and session history.

        Args:
            message: Original user message
            context: Optional additional context as key-value pairs
            memories: Optional list of memory strings
            session_history: Optional list of previous messages with 'role' and 'content'

        Returns:
            Composed message string with all sections
        """
        parts = []

        # Add session history if provided
        if session_history:
            history_text = self._format_session_history(session_history)
            parts.append(f"{self.HISTORY_HEADER}\n{history_text}\n")

        # Add memories if provided
        if memories:
            memory_text = self._format_memories(memories)
            parts.append(f"{self.MEMORIES_HEADER}\n{memory_text}\n")

        # Add context if provided
        if context:
            context_text = self._format_context(context)
            parts.append(f"{self.CONTEXT_HEADER}\n{context_text}\n")

        # Add the user message
        parts.append(message)

        return "\n".join(parts)

    def _format_session_history(self, history: list[dict[str, str]]) -> str:
        """Format session history into a string.

        Args:
            history: List of message dicts with 'role' and 'content' keys

        Returns:
            Formatted history string
        """
        return "\n".join(f"- {msg['role']}: {msg['content']}" for msg in history)

    def _format_memories(self, memories: list[str]) -> str:
        """Format memories into a string.

        Args:
            memories: List of memory strings

        Returns:
            Formatted memories string
        """
        return "\n".join(f"- {m}" for m in memories)

    def _format_context(self, context: dict[str, Any]) -> str:
        """Format context into a string.

        Args:
            context: Context dictionary

        Returns:
            Formatted context string
        """
        return "\n".join(f"- {k}: {v}" for k, v in context.items())

    @staticmethod
    def parse_session_events(events: list[Any]) -> list[dict[str, str]]:
        """Parse session events into message history format.

        Utility method for converting session events from the
        SessionManager into the format expected by build().

        Args:
            events: List of session event objects

        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        history = []
        for event in events:
            role = "user" if event.author.value == "user" else "assistant"
            content = event.content.get("text", str(event.content))
            history.append({"role": role, "content": content})
        return history
