# Dockerfile for Pydantic AI Agent Platform
# For local development and testing

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN pip install uv

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/

# Install dependencies using uv
RUN uv pip install --system -e ".[dev]"

# Copy remaining files
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY .env.example ./.env.example

# Create a non-root user
RUN useradd -m -u 1000 agent && chown -R agent:agent /app
USER agent

# Default command - run tests
CMD ["pytest", "tests/", "-v"]

# Alternative entrypoints:
# Run the agent locally:
#   docker run -it --env-file .env agent-engine python -c "from agent_engine import PydanticAIAgentWrapper; ..."
# Run specific tests:
#   docker run -it agent-engine pytest tests/test_tools.py -v
