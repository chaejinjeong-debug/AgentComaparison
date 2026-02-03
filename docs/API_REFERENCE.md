# API Reference

AgentEngine API 레퍼런스 문서

---

## Core Classes

### PydanticAIAgentWrapper

VertexAI Agent Engine 규격을 준수하는 Pydantic AI Agent Wrapper 클래스입니다.

```python
from agent_engine import PydanticAIAgentWrapper
```

#### Constructor

```python
PydanticAIAgentWrapper(
    model: str = "gemini-2.5-pro",
    project: str = "",
    location: str = "asia-northeast3",
    system_prompt: str = "You are a helpful AI assistant.",
    tools: Sequence[Callable[..., Any]] | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    config: AgentConfig | None = None,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"gemini-2.5-pro"` | Gemini 모델 이름 |
| `project` | `str` | `""` | GCP 프로젝트 ID |
| `location` | `str` | `"asia-northeast3"` | GCP 리전 |
| `system_prompt` | `str` | `"You are a helpful AI assistant."` | 시스템 프롬프트 |
| `tools` | `Sequence[Callable]` | `None` | 등록할 Tool 함수 목록 |
| `temperature` | `float` | `0.7` | 모델 온도 (0.0-2.0) |
| `max_tokens` | `int` | `4096` | 최대 출력 토큰 |
| `config` | `AgentConfig` | `None` | 전체 설정 (Phase 2+ 기능 사용 시) |

#### Class Methods

##### from_config

AgentConfig 인스턴스로부터 Agent Wrapper를 생성합니다.

```python
@classmethod
def from_config(
    cls,
    config: AgentConfig,
    tools: Sequence[Callable[..., Any]] | None = None,
) -> PydanticAIAgentWrapper
```

**Example:**

```python
config = AgentConfig(
    model="gemini-2.5-pro",
    project_id="my-project",
    location="asia-northeast3",
)
agent = PydanticAIAgentWrapper.from_config(config, tools=my_tools)
```

#### Instance Methods

##### set_up

Agent를 초기화합니다. 쿼리 실행 전 반드시 호출해야 합니다.

```python
def set_up(self) -> None
```

**Raises:**
- `AgentConfigError`: 초기화 실패 시

**Example:**

```python
agent = PydanticAIAgentWrapper(project="my-project")
agent.set_up()  # 반드시 호출
```

##### query

동기 쿼리를 실행합니다.

```python
def query(
    self,
    message: str,
    user_id: str | None = None,
    session_id: str | None = None,
    context: dict[str, Any] | None = None,
    memories: list[str] | None = None,
) -> dict[str, Any]
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `message` | `str` | Yes | 사용자 메시지 |
| `user_id` | `str` | No | 사용자 식별자 |
| `session_id` | `str` | No | 세션 ID |
| `context` | `dict` | No | 추가 컨텍스트 |
| `memories` | `list[str]` | No | 주입할 메모리 목록 |

**Returns:**

```python
{
    "response": str,          # Agent 응답 텍스트
    "tool_calls": [           # 실행된 Tool 목록
        {"tool": str, "args": dict}
    ],
    "usage": {                # 토큰 사용량
        "prompt_tokens": int,
        "completion_tokens": int,
        "total_tokens": int,
    },
    "metadata": {             # 메타데이터
        "model": str,
        "user_id": str | None,
        "session_id": str | None,
        "timestamp": str,
        "latency_ms": float,
    }
}
```

**Example:**

```python
response = agent.query(
    message="What time is it?",
    user_id="user123",
)
print(response["response"])
```

##### aquery

비동기 쿼리를 실행합니다.

```python
async def aquery(
    self,
    message: str,
    user_id: str | None = None,
    session_id: str | None = None,
    context: dict[str, Any] | None = None,
    memories: list[str] | None = None,
) -> dict[str, Any]
```

**Parameters:** `query()`와 동일

**Returns:** `query()`와 동일

**Example:**

```python
response = await agent.aquery(
    message="Calculate 15 * 23",
    user_id="user123",
)
```

##### stream_query

비동기 스트리밍 쿼리를 실행합니다.

```python
async def stream_query(
    self,
    message: str,
    user_id: str | None = None,
    session_id: str | None = None,
    context: dict[str, Any] | None = None,
    memories: list[str] | None = None,
) -> AsyncIterator[StreamChunk]
```

**Parameters:** `query()`와 동일

**Yields:** `StreamChunk` 객체

**Example:**

```python
async for chunk in agent.stream_query(message="Tell me a story"):
    if chunk.chunk:
        print(chunk.chunk, end="", flush=True)
    if chunk.done:
        print()  # 완료
        print(f"Latency: {chunk.metadata['latency_ms']}ms")
```

##### stream_query_sync

동기 스트리밍 쿼리를 실행합니다 (비동기 래퍼).

```python
def stream_query_sync(
    self,
    message: str,
    user_id: str | None = None,
    session_id: str | None = None,
    context: dict[str, Any] | None = None,
    memories: list[str] | None = None,
) -> Iterator[StreamChunk]
```

**Example:**

```python
for chunk in agent.stream_query_sync(message="Hello!"):
    print(chunk.chunk, end="", flush=True)
```

##### query_with_session

세션 관리가 통합된 비동기 쿼리를 실행합니다.

```python
async def query_with_session(
    self,
    message: str,
    user_id: str,
    session_id: str | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]
```

**Features:**
1. 세션 자동 생성/조회
2. 사용자 메시지 이벤트 기록
3. 관련 메모리 조회
4. 쿼리 실행
5. Agent 응답 이벤트 기록
6. 메모리 자동 생성 (설정 시)

**Returns:** `query()` 응답 + `session_id`

**Example:**

```python
# 첫 번째 메시지 - 새 세션 생성
response = await agent.query_with_session(
    message="My name is John",
    user_id="user123",
)
session_id = response["session_id"]

# 동일 세션에서 대화 계속
response = await agent.query_with_session(
    message="What's my name?",
    user_id="user123",
    session_id=session_id,
)
```

##### stream_query_with_session

세션 관리가 통합된 스트리밍 쿼리를 실행합니다.

```python
async def stream_query_with_session(
    self,
    message: str,
    user_id: str,
    session_id: str | None = None,
    context: dict[str, Any] | None = None,
) -> AsyncIterator[StreamChunk]
```

**Example:**

```python
async for chunk in agent.stream_query_with_session(
    message="Tell me more",
    user_id="user123",
    session_id=session_id,
):
    print(chunk.chunk, end="", flush=True)
```

##### add_tool

Tool을 Agent에 추가합니다. `set_up()` 전에 호출하세요.

```python
def add_tool(self, tool: Callable[..., Any]) -> None
```

##### register_tools

여러 Tool을 한 번에 등록합니다.

```python
def register_tools(self, tools: Sequence[Callable[..., Any]]) -> None
```

##### get_stats

Agent 통계를 반환합니다.

```python
def get_stats(self) -> dict[str, Any]
```

**Returns:**

```python
{
    "model": str,
    "is_setup": bool,
    "tool_count": int,
    "sessions": {...},  # SessionManager 통계 (활성화 시)
    "memory": {...},    # MemoryManager 통계 (활성화 시)
    "metrics": {...},   # MetricsManager 통계 (활성화 시)
}
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `session_manager` | `SessionManager | None` | 세션 매니저 인스턴스 |
| `memory_manager` | `MemoryManager | None` | 메모리 매니저 인스턴스 |

---

### StreamChunk

스트리밍 응답의 단일 청크를 나타내는 데이터클래스입니다.

```python
from agent_engine import StreamChunk
```

```python
@dataclass
class StreamChunk:
    chunk: str
    done: bool = False
    tool_call: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `chunk` | `str` | 텍스트 청크 |
| `done` | `bool` | 스트림 종료 여부 |
| `tool_call` | `dict | None` | Tool 호출 정보 (있는 경우) |
| `metadata` | `dict | None` | 메타데이터 (최종 청크에 포함) |

**Example:**

```python
async for chunk in agent.stream_query(message="Hello"):
    if chunk.tool_call:
        print(f"Tool called: {chunk.tool_call['tool']}")
    elif chunk.chunk:
        print(chunk.chunk, end="")
    elif chunk.done:
        print(f"\nCompleted in {chunk.metadata['latency_ms']}ms")
```

---

## Configuration Classes

### AgentConfig

Agent 전체 설정을 담는 클래스입니다.

```python
from agent_engine import AgentConfig
```

```python
@dataclass
class AgentConfig:
    model: str = "gemini-2.5-pro"
    project_id: str = ""
    location: str = "asia-northeast3"
    system_prompt: str = "You are a helpful AI assistant."
    temperature: float = 0.7
    max_tokens: int = 4096
    log_level: str = "INFO"
    log_format: str = "json"
    session: SessionConfig = field(default_factory=SessionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
```

---

### SessionConfig

세션 관리 설정입니다.

```python
from agent_engine import SessionConfig, SessionBackendType
```

```python
@dataclass
class SessionConfig:
    enabled: bool = False
    backend_type: SessionBackendType = SessionBackendType.IN_MEMORY
    ttl_hours: int = 24
    agent_engine_id: str | None = None  # VertexAI 백엔드용
```

**SessionBackendType:**

| Value | Description |
|-------|-------------|
| `IN_MEMORY` | 메모리 내 저장 (개발/테스트용) |
| `VERTEX_AI` | VertexAI Agent Engine Sessions API |

---

### MemoryConfig

메모리 관리 설정입니다.

```python
from agent_engine import MemoryConfig, MemoryBackendType
```

```python
@dataclass
class MemoryConfig:
    enabled: bool = False
    backend_type: MemoryBackendType = MemoryBackendType.IN_MEMORY
    auto_generate: bool = False
    max_memories_per_user: int = 100
    agent_engine_id: str | None = None  # VertexAI 백엔드용
```

**MemoryBackendType:**

| Value | Description |
|-------|-------------|
| `IN_MEMORY` | 메모리 내 저장 (개발/테스트용) |
| `VERTEX_AI` | VertexAI Agent Engine Memory Bank API |

---

### ObservabilityConfig

관측성 설정입니다.

```python
from agent_engine import ObservabilityConfig
```

```python
@dataclass
class ObservabilityConfig:
    tracing_enabled: bool = False
    logging_enabled: bool = True
    metrics_enabled: bool = False
    trace_sample_rate: float = 1.0
```

---

## Exception Classes

### AgentError

모든 Agent 예외의 기본 클래스입니다.

```python
from agent_engine import AgentError
```

### AgentConfigError

설정 관련 예외입니다.

```python
from agent_engine import AgentConfigError
```

**Example:**

```python
try:
    agent.query(message="Hello")  # set_up() 호출 전
except AgentConfigError as e:
    print(f"Config error: {e}")
```

### AgentQueryError

쿼리 실행 중 예외입니다.

```python
from agent_engine import AgentQueryError
```

**Attributes:**
- `user_id`: 사용자 ID
- `session_id`: 세션 ID
- `details`: 추가 상세 정보

### ToolExecutionError

Tool 실행 중 예외입니다.

```python
from agent_engine import ToolExecutionError
```

---

## Session Management

### SessionManager

세션 관리를 담당하는 클래스입니다.

```python
from agent_engine.sessions import SessionManager
```

#### Methods

##### create_session

```python
async def create_session(
    self,
    user_id: str,
    metadata: dict[str, Any] | None = None,
) -> Session
```

##### get_session

```python
async def get_session(self, session_id: str) -> Session | None
```

##### delete_session

```python
async def delete_session(self, session_id: str) -> bool
```

##### append_event

```python
async def append_event(
    self,
    session_id: str,
    author: EventAuthor,
    content: dict[str, Any],
) -> SessionEvent
```

##### list_events

```python
async def list_events(self, session_id: str) -> list[SessionEvent]
```

---

### Session

세션 모델입니다.

```python
from agent_engine.sessions import Session, SessionStatus
```

```python
@dataclass
class Session:
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    status: SessionStatus
    metadata: dict[str, Any]
```

---

## Memory Management

### MemoryManager

메모리 관리를 담당하는 클래스입니다.

```python
from agent_engine.memory import MemoryManager
```

#### Methods

##### create_memory

```python
async def create_memory(
    self,
    user_id: str,
    fact: str,
    source: str = "agent",
) -> Memory
```

##### retrieve_memories

```python
async def retrieve_memories(
    self,
    user_id: str,
    query: str | None = None,
    max_results: int = 10,
) -> list[Memory]
```

##### delete_memory

```python
async def delete_memory(self, memory_id: str) -> bool
```

##### purge_user_memories

```python
async def purge_user_memories(self, user_id: str) -> int
```

---

### Memory

메모리 모델입니다.

```python
from agent_engine.memory import Memory
```

```python
@dataclass
class Memory:
    memory_id: str
    user_id: str
    fact: str
    created_at: datetime
    source: str
```

---

## Version Management

### VersionRegistry

버전 레지스트리를 관리합니다.

```python
from agent_engine import VersionRegistry
```

#### Methods

##### register_version

```python
def register_version(
    self,
    version: str,
    environment: str,
    agent_engine_id: str,
    deployment_info: DeploymentInfo,
) -> Version
```

##### get_current_version

```python
def get_current_version(self, environment: str) -> Version | None
```

##### list_versions

```python
def list_versions(self, environment: str) -> list[Version]
```

---

### RollbackManager

롤백을 관리합니다.

```python
from agent_engine import RollbackManager
```

#### Methods

##### rollback

```python
def rollback(
    self,
    environment: str,
    target_version: str | None = None,
    reason: str = "",
) -> Version
```

---

## Built-in Tools

### DEFAULT_TOOLS

기본 제공 Tool 목록입니다.

```python
from agent_engine.tools import DEFAULT_TOOLS

# DEFAULT_TOOLS 포함:
# - search_knowledge: 지식 검색
# - calculate: 수식 계산
# - get_current_datetime: 현재 날짜/시간
```

### Memory Tools

```python
from agent_engine.tools.memory_tools import save_memory, recall_user_info
```

**save_memory**: 메모리를 저장합니다.
**recall_user_info**: 사용자 정보를 조회합니다.

---

## Observability

### TracingManager

Cloud Trace 통합을 관리합니다.

```python
from agent_engine.observability import TracingManager
```

### LoggingManager

구조화된 로깅을 관리합니다.

```python
from agent_engine.observability import LoggingManager
```

### MetricsManager

메트릭 수집을 관리합니다.

```python
from agent_engine.observability import MetricsManager
```

### Decorators

```python
from agent_engine.observability.decorators import traced, metered, logged_tool
```

**@traced**: 함수에 트레이싱을 추가합니다.
**@metered**: 함수에 메트릭을 추가합니다.
**@logged_tool**: Tool 실행 로깅을 추가합니다.

---

## Related Documentation

- [README.md](../README.md) - 프로젝트 개요
- [DEPLOYMENT.md](./DEPLOYMENT.md) - 배포 가이드
- [SECURITY.md](./SECURITY.md) - 보안 가이드
- [RUNBOOK.md](./RUNBOOK.md) - 운영 런북
