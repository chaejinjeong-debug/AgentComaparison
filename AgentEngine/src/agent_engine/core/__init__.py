"""Core module for Agent Engine.

This module provides the foundational classes for building and deploying
agents with separation of concerns:

- BaseAgent: Abstract interface for all agent implementations
- PydanticAIAgent: Pydantic AI specific implementation
- AgentEngineWrapper: Deployment wrapper for Agent Engine spec
- MessageBuilder: Message composition utility
- ResultProcessor: Result processing utility
"""

from agent_engine.core.base_agent import AgentResult, BaseAgent
from agent_engine.core.message_builder import MessageBuilder
from agent_engine.core.pydantic_agent import PydanticAIAgent
from agent_engine.core.result_processor import ResultProcessor
from agent_engine.core.wrapper import AgentEngineWrapper, StreamChunk

__all__ = [
    # Abstract interface
    "BaseAgent",
    "AgentResult",
    # Implementations
    "PydanticAIAgent",
    "AgentEngineWrapper",
    # Utilities
    "MessageBuilder",
    "ResultProcessor",
    "StreamChunk",
]
