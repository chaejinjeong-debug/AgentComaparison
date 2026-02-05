#!/usr/bin/env python3
"""Deploy agent with session-based memory support.

This script deploys a Gemini agent that uses VertexAI Sessions API
to maintain conversation context across queries.

Usage:
    cd AgentEngine && uv run python scripts/deploy_with_session.py

Requirements:
    - GCP authentication: gcloud auth application-default login
    - Staging bucket exists or will be created
"""

import argparse
import sys
from pathlib import Path

import structlog

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class SessionAwareAgent:
    """Gemini agent with session-based memory support.

    This agent uses VertexAI Sessions API to maintain conversation
    context across multiple queries.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        project: str = "",
        location: str = "asia-northeast3",
        system_prompt: str = "You are a helpful AI assistant. Use the conversation history to maintain context.",
    ) -> None:
        self.model = model
        self.project = project
        self.location = location
        self.system_prompt = system_prompt
        self._client = None
        self._vertex_client = None
        self._agent_engine_name = None

    def set_up(self) -> None:
        """Initialize the Gemini client and VertexAI SDK."""
        import vertexai
        from vertexai.generative_models import GenerativeModel

        vertexai.init(project=self.project, location=self.location)
        self._client = GenerativeModel(
            model_name=self.model,
            system_instruction=self.system_prompt,
        )

        # Initialize VertexAI SDK client for Sessions API
        self._vertex_client = vertexai.Client(
            project=self.project,
            location=self.location,
        )

    def _get_or_create_session(
        self, user_id: str, session_id: str | None = None
    ) -> tuple[str, str]:
        """Get existing session or create a new one.

        Args:
            user_id: User identifier
            session_id: Optional existing session ID

        Returns:
            Tuple of (session_name, session_id)
        """
        import datetime

        if self._agent_engine_name is None:
            # Get the agent engine name from environment or construct it
            import os

            agent_engine_id = os.environ.get("AGENT_ENGINE_ID", "")
            if agent_engine_id:
                self._agent_engine_name = (
                    f"projects/{self.project}/locations/{self.location}"
                    f"/reasoningEngines/{agent_engine_id}"
                )
            else:
                # Skip session management if no agent engine ID
                return "", ""

        if session_id:
            session_name = f"{self._agent_engine_name}/sessions/{session_id}"
            return session_name, session_id

        # Create new session
        try:
            expire_time = datetime.datetime.now(tz=datetime.UTC) + datetime.timedelta(hours=25)
            result = self._vertex_client.agent_engines.sessions.create(
                name=self._agent_engine_name,
                user_id=user_id,
                config={"expire_time": expire_time},
            )

            session_name = result.name
            if "/operations/" in session_name:
                session_name = session_name.split("/operations/")[0]

            session_id = session_name.split("/sessions/")[-1]
            return session_name, session_id
        except Exception as e:
            print(f"Warning: Could not create session: {e}")
            return "", ""

    def _get_session_history(self, session_name: str) -> list[dict]:
        """Retrieve session history from VertexAI Sessions API.

        Args:
            session_name: Full session resource name

        Returns:
            List of message dicts with 'role' and 'content'
        """
        if not session_name or not self._vertex_client:
            return []

        try:
            events = list(self._vertex_client.agent_engines.sessions.events.list(name=session_name))
            history = []
            for event in events:
                author = getattr(event, "author", "user")
                role = "user" if author == "user" else "assistant"

                content = getattr(event, "content", {})
                if hasattr(content, "parts"):
                    parts = content.parts
                    text = parts[0].text if parts else ""
                elif isinstance(content, dict):
                    parts = content.get("parts", [])
                    text = parts[0].get("text", "") if parts else ""
                else:
                    text = str(content)

                if text:
                    history.append({"role": role, "content": text})

            return history
        except Exception as e:
            print(f"Warning: Could not retrieve session history: {e}")
            return []

    def _append_event(self, session_name: str, author: str, text: str, invocation_id: str) -> None:
        """Append an event to the session.

        Args:
            session_name: Full session resource name
            author: Event author ('user' or 'agent')
            text: Event text content
            invocation_id: Unique invocation ID
        """
        if not session_name or not self._vertex_client:
            return

        import datetime

        try:
            role = "user" if author == "user" else "model"
            self._vertex_client.agent_engines.sessions.events.append(
                name=session_name,
                author=author,
                invocation_id=invocation_id,
                timestamp=datetime.datetime.now(tz=datetime.UTC),
                config={"content": {"role": role, "parts": [{"text": text}]}},
            )
        except Exception as e:
            print(f"Warning: Could not append event: {e}")

    def _build_prompt_with_history(self, message: str, history: list[dict]) -> str:
        """Build prompt with conversation history.

        Args:
            message: Current user message
            history: Conversation history

        Returns:
            Full prompt string
        """
        if not history:
            return message

        history_text = "\n".join(f"- {msg['role']}: {msg['content']}" for msg in history)

        return f"""[Previous conversation]
{history_text}

[Current message]
{message}"""

    def query(
        self,
        *,
        message: str,
        user_id: str | None = None,
        session_id: str | None = None,
        **kwargs,
    ) -> dict:
        """Query the agent with session-based memory.

        Args:
            message: User message
            user_id: Optional user identifier for session
            session_id: Optional existing session ID

        Returns:
            Response dict with response text and session_id
        """
        from datetime import UTC, datetime
        from uuid import uuid4

        if self._client is None:
            self.set_up()

        start_time = datetime.now(UTC)
        invocation_id = str(uuid4())

        # Get or create session
        session_name = ""
        new_session_id = session_id
        if user_id:
            session_name, new_session_id = self._get_or_create_session(user_id, session_id)

        # Get session history
        history = self._get_session_history(session_name) if session_name else []

        # Append user message to session
        if session_name:
            self._append_event(session_name, "user", message, invocation_id)

        # Build prompt with history
        full_prompt = self._build_prompt_with_history(message, history)

        try:
            response = self._client.generate_content(full_prompt)
            response_text = response.text

            # Append agent response to session
            if session_name:
                self._append_event(session_name, "agent", response_text, invocation_id)

            result = {
                "response": response_text,
                "metadata": {
                    "model": self.model,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "latency_ms": (datetime.now(UTC) - start_time).total_seconds() * 1000,
                },
            }

            if new_session_id:
                result["session_id"] = new_session_id

            return result

        except Exception as e:
            return {
                "response": f"Error: {e}",
                "metadata": {"error": str(e)},
            }


def deploy_agent(
    project: str,
    location: str,
    display_name: str,
    staging_bucket: str | None = None,
) -> str:
    """Deploy the session-aware agent to VertexAI."""
    import vertexai
    from vertexai.preview.reasoning_engines import ReasoningEngine

    logger.info(
        "starting_deployment",
        project=project,
        location=location,
        display_name=display_name,
    )

    if not staging_bucket:
        staging_bucket = f"gs://{project}-agent-staging-{location.replace('-', '')}"

    vertexai.init(project=project, location=location, staging_bucket=staging_bucket)

    agent = SessionAwareAgent(
        model="gemini-2.5-flash",
        project=project,
        location=location,
    )

    requirements = [
        "google-cloud-aiplatform[agent_engines]>=1.78.0",
    ]

    deployed_agent = ReasoningEngine.create(
        reasoning_engine=agent,
        requirements=requirements,
        display_name=display_name,
        description="Gemini agent with session-based memory support",
    )

    agent_name = deployed_agent.resource_name
    agent_id = agent_name.split("/")[-1]

    logger.info(
        "deployment_complete",
        agent_name=agent_name,
        agent_id=agent_id,
    )

    return agent_name


def main():
    import os

    from dotenv import load_dotenv

    # Load .env from project root
    load_dotenv(Path(__file__).parent.parent / ".env")

    parser = argparse.ArgumentParser(description="Deploy session-aware agent")
    parser.add_argument("--project", default=os.getenv("AGENT_PROJECT_ID", ""))
    parser.add_argument("--location", default=os.getenv("AGENT_LOCATION", "asia-northeast3"))
    parser.add_argument("--display-name", default="session-aware-agent")
    parser.add_argument("--staging-bucket", default=None)

    args = parser.parse_args()

    print("=" * 60)
    print("Deploying Session-Aware Agent")
    print("=" * 60)
    print(f"Project: {args.project}")
    print(f"Location: {args.location}")
    print(f"Display Name: {args.display_name}")
    print("=" * 60 + "\n")

    try:
        agent_name = deploy_agent(
            project=args.project,
            location=args.location,
            display_name=args.display_name,
            staging_bucket=args.staging_bucket,
        )

        agent_id = agent_name.split("/")[-1]

        print("\n" + "=" * 60)
        print("Deployment Successful!")
        print("=" * 60)
        print(f"Agent Name: {agent_name}")
        print(f"Agent ID: {agent_id}")
        print("\nTo test:")
        print(f"  Update AGENT_ENGINE_ID in test scripts to: {agent_id}")
        print("=" * 60)

    except Exception as e:
        print(f"\nDeployment failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
