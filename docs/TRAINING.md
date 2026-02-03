# Training Guide

AgentEngine 팀 교육 자료

---

## 1. 개요

### 1.1 AgentEngine이란?

AgentEngine은 VertexAI Agent Engine 위에 Pydantic AI를 기반으로 구축된 Production-ready AI Agent 플랫폼입니다.

```
┌─────────────────────────────────────────────────────────┐
│                    사용자 애플리케이션                     │
├─────────────────────────────────────────────────────────┤
│                     AgentEngine                          │
│  ┌─────────────┬─────────────┬─────────────────────┐   │
│  │   Agent     │   Session   │      Memory         │   │
│  │   Wrapper   │   Manager   │      Bank           │   │
│  └─────────────┴─────────────┴─────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│                  VertexAI Agent Engine                   │
│           (Runtime, Sessions, Memory Bank)               │
├─────────────────────────────────────────────────────────┤
│                      Gemini API                          │
└─────────────────────────────────────────────────────────┘
```

### 1.2 핵심 기능

| 기능 | 설명 | Phase |
|------|------|-------|
| Agent Wrapper | Pydantic AI Agent를 Agent Engine 규격으로 래핑 | 1 |
| Tool System | @tool 데코레이터 기반 Tool 등록 | 1 |
| Session Management | 대화 컨텍스트 유지 | 2 |
| Memory Bank | 장기 사용자 메모리 저장/조회 | 2 |
| Observability | Tracing, Logging, Metrics | 2 |
| Version Management | 배포 버전 관리 및 롤백 | 3 |
| Streaming | 실시간 응답 스트리밍 | 4 |

---

## 2. 개발 환경 설정

### 2.1 필수 도구

```bash
# Python 3.11+
python --version  # Python 3.11.x

# uv 패키지 매니저
curl -LsSf https://astral.sh/uv/install.sh | sh

# gcloud CLI
gcloud --version
```

### 2.2 프로젝트 설정

```bash
# 저장소 클론
git clone <repository-url>
cd AgentEngine

# 의존성 설치
uv sync --all-extras

# 환경 변수 설정
cp .env.example .env
# .env 파일 편집

# GCP 인증
gcloud auth application-default login
```

### 2.3 IDE 설정

**VS Code 권장 익스텐션:**
- Python
- Pylance
- Ruff
- Even Better TOML

**settings.json:**
```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.analysis.typeCheckingMode": "basic",
  "[python]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll": true
    }
  }
}
```

---

## 3. 기본 사용법

### 3.1 Agent 생성 및 쿼리

```python
from agent_engine import PydanticAIAgentWrapper
from agent_engine.tools import DEFAULT_TOOLS

# 1. Agent 생성
agent = PydanticAIAgentWrapper(
    model="gemini-2.5-pro",
    project="my-project-id",
    location="asia-northeast3",
    system_prompt="You are a helpful assistant.",
    tools=DEFAULT_TOOLS,
)

# 2. 초기화 (필수!)
agent.set_up()

# 3. 쿼리 실행
response = agent.query(
    message="What time is it?",
    user_id="user123",
)

print(response["response"])
print(f"Tokens used: {response['usage']['total_tokens']}")
```

### 3.2 비동기 쿼리

```python
import asyncio

async def main():
    agent = PydanticAIAgentWrapper(
        model="gemini-2.5-pro",
        project="my-project-id",
    )
    agent.set_up()

    # 비동기 쿼리
    response = await agent.aquery(
        message="Calculate 15 * 23",
        user_id="user123",
    )
    print(response["response"])

asyncio.run(main())
```

### 3.3 스트리밍 응답

```python
async def stream_example():
    agent = PydanticAIAgentWrapper(project="my-project-id")
    agent.set_up()

    print("Agent: ", end="")
    async for chunk in agent.stream_query(message="Tell me a story"):
        if chunk.chunk:
            print(chunk.chunk, end="", flush=True)
        if chunk.done:
            print()  # 줄바꿈
            print(f"\n[Completed in {chunk.metadata['latency_ms']:.0f}ms]")
```

---

## 4. Tool 개발

### 4.1 기본 Tool 작성

```python
from pydantic_ai import tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city.

    Args:
        city: The city name to get weather for.

    Returns:
        Weather information as a string.
    """
    # 실제 구현에서는 외부 API 호출
    return f"Weather in {city}: Sunny, 22°C"

# Agent에 Tool 등록
agent = PydanticAIAgentWrapper(
    project="my-project-id",
    tools=[get_weather],
)
```

### 4.2 비동기 Tool

```python
import httpx
from pydantic_ai import tool

@tool
async def fetch_stock_price(symbol: str) -> str:
    """Fetch current stock price.

    Args:
        symbol: Stock ticker symbol (e.g., AAPL, GOOGL)

    Returns:
        Current stock price information.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.example.com/stocks/{symbol}")
        data = response.json()
        return f"{symbol}: ${data['price']}"
```

### 4.3 Tool 테스트

```python
import pytest

def test_get_weather():
    result = get_weather("Seoul")
    assert "Seoul" in result
    assert "Weather" in result

@pytest.mark.asyncio
async def test_fetch_stock_price():
    result = await fetch_stock_price("AAPL")
    assert "AAPL" in result
```

---

## 5. Session 관리

### 5.1 Session 활성화

```python
from agent_engine import AgentConfig, SessionConfig

config = AgentConfig(
    model="gemini-2.5-pro",
    project_id="my-project-id",
    session=SessionConfig(
        enabled=True,
        ttl_hours=24,  # 세션 만료 시간
    ),
)

agent = PydanticAIAgentWrapper.from_config(config)
agent.set_up()
```

### 5.2 Session 기반 대화

```python
async def conversation_example():
    # 첫 번째 메시지 - 새 세션 생성
    response1 = await agent.query_with_session(
        message="My name is John and I like Python.",
        user_id="user123",
    )
    session_id = response1["session_id"]
    print(f"Session created: {session_id}")

    # 두 번째 메시지 - 동일 세션
    response2 = await agent.query_with_session(
        message="What's my name and what do I like?",
        user_id="user123",
        session_id=session_id,
    )
    print(response2["response"])
    # "Your name is John and you like Python."
```

### 5.3 Session 직접 관리

```python
from agent_engine.sessions import EventAuthor

session_manager = agent.session_manager

# 세션 생성
session = await session_manager.create_session(user_id="user123")

# 이벤트 추가
await session_manager.append_event(
    session_id=session.session_id,
    author=EventAuthor.USER,
    content={"text": "Hello!"},
)

# 이벤트 조회
events = await session_manager.list_events(session.session_id)
for event in events:
    print(f"{event.author.value}: {event.content}")

# 세션 삭제
await session_manager.delete_session(session.session_id)
```

---

## 6. Memory Bank

### 6.1 Memory 활성화

```python
from agent_engine import AgentConfig, MemoryConfig

config = AgentConfig(
    model="gemini-2.5-pro",
    project_id="my-project-id",
    memory=MemoryConfig(
        enabled=True,
        auto_generate=True,  # 대화에서 자동 fact 추출
        max_memories_per_user=100,
    ),
)
```

### 6.2 Memory 직접 관리

```python
memory_manager = agent.memory_manager

# 메모리 생성
memory = await memory_manager.create_memory(
    user_id="user123",
    fact="User prefers email communication",
)

# 메모리 조회 (Similarity Search)
memories = await memory_manager.retrieve_memories(
    user_id="user123",
    query="How does the user prefer to be contacted?",
    max_results=5,
)
for m in memories:
    print(f"- {m.fact}")

# 사용자 메모리 삭제 (GDPR 대응)
count = await memory_manager.purge_user_memories(user_id="user123")
print(f"Deleted {count} memories")
```

### 6.3 Memory Tool 사용

```python
from agent_engine.tools.memory_tools import save_memory, recall_user_info

# Agent가 명시적으로 메모리 저장
agent = PydanticAIAgentWrapper(
    project="my-project-id",
    tools=[save_memory, recall_user_info],
    system_prompt="""
    You can save important facts about users using save_memory tool.
    You can recall user information using recall_user_info tool.
    """,
)
```

---

## 7. Observability

### 7.1 활성화

```python
from agent_engine import AgentConfig, ObservabilityConfig

config = AgentConfig(
    model="gemini-2.5-pro",
    project_id="my-project-id",
    observability=ObservabilityConfig(
        tracing_enabled=True,   # Cloud Trace
        logging_enabled=True,   # Structured Logging
        metrics_enabled=True,   # Cloud Monitoring
    ),
)
```

### 7.2 로그 확인

```bash
# 최근 로그 조회
gcloud logging read \
  'resource.type="aiplatform.googleapis.com/AgentEngine"' \
  --limit=10 \
  --format=json

# 에러 로그만
gcloud logging read \
  'severity>=ERROR' \
  --limit=10
```

### 7.3 트레이스 확인

Cloud Console > Trace 에서 요청 추적 가능

### 7.4 메트릭 확인

Cloud Console > Monitoring > Metrics Explorer

주요 메트릭:
- `agent_requests_total`: 요청 수
- `agent_latency_seconds`: 지연 시간
- `agent_errors_total`: 에러 수
- `agent_token_usage`: 토큰 사용량

---

## 8. 테스트

### 8.1 단위 테스트

```bash
# 전체 테스트
uv run pytest tests/ -v

# 특정 파일
uv run pytest tests/test_agent.py -v

# 특정 테스트
uv run pytest tests/test_agent.py::TestAgentQuery::test_query_success -v

# 커버리지
uv run pytest tests/ --cov=agent_engine --cov-report=html
```

### 8.2 테스트 작성

```python
import pytest
from unittest.mock import MagicMock, patch

from agent_engine import PydanticAIAgentWrapper
from agent_engine.exceptions import AgentQueryError

class TestMyFeature:
    """Test suite for my feature."""

    def test_basic_functionality(self):
        """Test basic functionality."""
        # Arrange
        agent = PydanticAIAgentWrapper(project="test")

        # Act & Assert
        with pytest.raises(AgentConfigError):
            agent.query(message="Hello")  # set_up 안 함

    @pytest.mark.asyncio
    async def test_async_functionality(self):
        """Test async functionality."""
        # async 테스트 코드
        pass

    @patch("vertexai.init")
    @patch("pydantic_ai.Agent")
    def test_with_mocks(self, mock_agent, mock_vertexai):
        """Test with mocks."""
        # 목 설정
        mock_agent.return_value.run_sync.return_value = MagicMock()

        # 테스트 코드
        pass
```

### 8.3 통합 테스트

```python
@pytest.mark.integration
async def test_full_conversation_flow():
    """Test complete conversation with sessions and memory."""
    config = AgentConfig(
        project_id="test-project",
        session=SessionConfig(enabled=True),
        memory=MemoryConfig(enabled=True),
    )
    agent = PydanticAIAgentWrapper.from_config(config)
    agent.set_up()

    # 대화 플로우 테스트
    response1 = await agent.query_with_session(
        message="I'm Alice",
        user_id="test-user",
    )
    assert response1["session_id"]

    response2 = await agent.query_with_session(
        message="What's my name?",
        user_id="test-user",
        session_id=response1["session_id"],
    )
    assert "Alice" in response2["response"]
```

---

## 9. 배포

### 9.1 로컬 테스트 → Staging

```bash
# 1. 코드 품질 확인
uv run ruff check src/ tests/
uv run mypy src/

# 2. 테스트 실행
uv run pytest tests/ -v

# 3. Staging 배포
uv run python scripts/deploy_source.py --env staging
```

### 9.2 Staging → Production

```bash
# 1. Staging에서 테스트
uv run python scripts/chat_with_agent.py --env staging

# 2. 버전 태그 생성
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0

# 3. Production 배포 (자동 또는 수동)
uv run python scripts/deploy_source.py --env production
```

### 9.3 롤백

```bash
# 이전 버전으로 롤백
uv run python scripts/version/rollback.py \
  --env production \
  --reason "Bug in v1.1.0"

# 특정 버전으로 롤백
uv run python scripts/version/rollback.py \
  --env production \
  --target v1.0.0 \
  --reason "Rollback to stable"
```

---

## 10. 문제 해결

### 10.1 자주 발생하는 오류

**AgentConfigError: Agent not set up**
```python
# 원인: set_up() 호출 전 쿼리 시도
# 해결:
agent = PydanticAIAgentWrapper(project="my-project")
agent.set_up()  # 반드시 호출!
response = agent.query(message="Hello")
```

**ImportError: No module named 'vertexai'**
```bash
# 해결:
uv sync --all-extras
```

**Permission denied**
```bash
# GCP 인증 확인
gcloud auth application-default login

# 또는 서비스 계정
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
```

### 10.2 디버깅 팁

```python
# 상세 로깅 활성화
import logging
logging.basicConfig(level=logging.DEBUG)

# Agent 상태 확인
print(agent.get_stats())

# Tool 목록 확인
print([t.__name__ for t in agent.tools])
```

### 10.3 도움말

- [API Reference](./API_REFERENCE.md)
- [Deployment Guide](./DEPLOYMENT.md)
- [Operations Runbook](./RUNBOOK.md)
- Slack: #agent-engine-support

---

## 11. 연습 문제

### 연습 1: 기본 Agent

날씨 정보를 제공하는 Agent를 만들어보세요.

1. `get_weather` Tool 구현
2. Agent 생성 및 설정
3. 쿼리 테스트

### 연습 2: Session 활용

다음 기능을 구현하세요:

1. Session 활성화
2. 사용자 이름 기억
3. 이전 대화 참조

### 연습 3: Memory 활용

장기 메모리를 활용한 Agent를 만드세요:

1. Memory 활성화
2. 사용자 선호도 저장
3. 이후 세션에서 활용

### 연습 4: Custom Tool

외부 API를 호출하는 Tool을 만드세요:

1. 비동기 Tool 구현
2. 에러 처리
3. 단위 테스트 작성

---

## 12. 추가 자료

### 문서

- [VertexAI Agent Engine 공식 문서](https://cloud.google.com/agent-builder/agent-engine)
- [Pydantic AI 공식 문서](https://ai.pydantic.dev/)
- [PRD.md](../PRD.md) - 프로젝트 요구사항

### 코드 예제

- `examples/` 디렉토리 참조
- `tests/` 디렉토리의 테스트 코드 참조

### 커뮤니티

- GitHub Issues: 버그 리포트 및 기능 요청
- Slack: #agent-engine-dev
