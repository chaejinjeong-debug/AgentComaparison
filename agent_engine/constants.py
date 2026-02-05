"""Constants for the Agent Engine platform.

This module centralizes all default values and constants used across
the platform to ensure consistency and maintainability.
"""

# =============================================================================
# Model Settings
# =============================================================================
DEFAULT_MODEL = "gemini-2.5-pro"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4096
DEFAULT_LOCATION = "asia-northeast3"

# =============================================================================
# Session Settings
# =============================================================================
DEFAULT_SESSION_TTL_SECONDS = 86400  # 24 hours
DEFAULT_MAX_EVENTS_PER_SESSION = 1000

# =============================================================================
# Memory Settings
# =============================================================================
DEFAULT_MAX_MEMORIES_PER_USER = 1000
DEFAULT_SIMILARITY_THRESHOLD = 0.7

# =============================================================================
# Evaluation Settings
# =============================================================================
DEFAULT_QUALITY_THRESHOLD = 0.85
DEFAULT_P50_THRESHOLD_MS = 2000.0
DEFAULT_P99_THRESHOLD_MS = 10000.0
DEFAULT_ERROR_RATE_THRESHOLD = 0.05
DEFAULT_EVALUATION_TIMEOUT_SECONDS = 30.0
DEFAULT_TEST_DATA_PATH = "tests/evaluation/golden/qa_pairs.json"

# =============================================================================
# Agent Settings
# =============================================================================
DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."
DEFAULT_DISPLAY_NAME = "pydantic-ai-agent"
DEFAULT_DESCRIPTION = "Pydantic AI Agent on VertexAI Agent Engine"

# =============================================================================
# Logging Settings
# =============================================================================
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "json"
DEFAULT_TRACE_SAMPLE_RATE = 1.0

# =============================================================================
# Valid Values
# =============================================================================
VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
VALID_LOG_FORMATS = {"json", "text"}
