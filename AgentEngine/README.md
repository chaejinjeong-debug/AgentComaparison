# Pydantic AI Agent Platform on VertexAI Agent Engine

VertexAI Agent Engine에 배포 가능한 Pydantic AI 기반 Production-ready Agent 플랫폼입니다.

## Features

### Core Features (Phase 1)
- **Agent Engine 규격 준수**: `__init__`, `set_up`, `query` 메서드 구현
- **Pydantic AI 통합**: Pydantic AI Agent를 Agent Engine에서 실행
- **Gemini 모델 연동**: GoogleProvider를 통한 VertexAI Gemini 모델 사용
- **동기/비동기 지원**: `query` (sync), `aquery` (async) 메서드 제공
- **기본 Tool 라이브러리**: 검색, 계산, 날짜/시간 Tool 포함
- **Docker 지원**: 로컬 개발/테스트 환경 제공

### Session & Memory (Phase 2)
- **Session 관리**: 대화 컨텍스트 유지 (생성, 이벤트 기록, 조회, 삭제)
- **Memory Bank**: 장기 메모리 저장/조회 (Similarity Search 지원)
- **Observability**: Cloud Trace, Logging, Metrics 통합
- **모니터링 대시보드**: 주요 지표 시각화

### Production Readiness (Phase 3)
- **CI/CD 파이프라인**: Cloud Build 기반 자동화
- **버전 관리**: 배포 버전 관리 및 롤백 지원
- **환경 분리**: Staging/Production 환경 분리
- **Agent 평가**: 품질 평가 자동화 (85% 임계치)
- **보안**: VPC-SC, IAM 최소 권한, CMEK 지원

### Advanced Features (Phase 4)
- **스트리밍 응답**: `stream_query`, `astream_query` 메서드
- **세션 기반 스트리밍**: `stream_query_with_session` 메서드

## Requirements

- Python 3.11+
- uv 패키지 매니저
- GCP 프로젝트 및 인증

## Installation

```bash
# uv 설치 (없는 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치
uv sync

# 개발 의존성 포함
uv sync --all-extras
```

## Quick Start

### 1. 환경 설정

```bash
cp .env.example .env
# .env 파일을 편집하여 GCP 프로젝트 ID 등 설정
```

### 2. 기본 사용법

```python
from agent_engine import PydanticAIAgentWrapper, AgentConfig
from agent_engine.tools import DEFAULT_TOOLS

# Agent 생성
agent = PydanticAIAgentWrapper(
    model="gemini-2.5-pro",
    project="your-project-id",
    location="asia-northeast3",
    system_prompt="You are a helpful assistant.",
    tools=DEFAULT_TOOLS,
)

# 초기화
agent.set_up()

# 동기 쿼리
response = agent.query(
    message="What time is it in Seoul?",
    user_id="user123",
)
print(response["response"])

# 비동기 쿼리
response = await agent.aquery(
    message="Calculate 15 * 23",
    user_id="user123",
)
```

### 3. 스트리밍 응답

```python
# 비동기 스트리밍
async for chunk in agent.stream_query(message="Tell me a story"):
    print(chunk.chunk, end="", flush=True)
    if chunk.done:
        print()  # Final newline

# 동기 스트리밍
for chunk in agent.stream_query_sync(message="Hello!"):
    print(chunk.chunk, end="", flush=True)
```

### 4. Session 기반 대화

```python
from agent_engine import AgentConfig, SessionConfig, MemoryConfig

# Config로 Agent 생성
config = AgentConfig(
    model="gemini-2.5-pro",
    project_id="your-project-id",
    location="asia-northeast3",
    session=SessionConfig(enabled=True, ttl_hours=24),
    memory=MemoryConfig(enabled=True, auto_generate=True),
)

agent = PydanticAIAgentWrapper.from_config(config, tools=DEFAULT_TOOLS)
agent.set_up()

# 세션 기반 쿼리 (대화 기록 자동 관리)
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
# Agent가 "John"을 기억함

# 세션 기반 스트리밍
async for chunk in agent.stream_query_with_session(
    message="Tell me more",
    user_id="user123",
    session_id=session_id,
):
    print(chunk.chunk, end="", flush=True)
```

### 5. 로컬 테스트

```bash
# 테스트 실행
uv run pytest tests/ -v

# 커버리지 포함
uv run pytest tests/ -v --cov=agent_engine --cov-report=html

# 린팅
uv run ruff check src/ tests/

# 타입 체크
uv run mypy src/
```

### 6. Agent Engine 배포

```bash
# Staging 배포
uv run python scripts/deploy_source.py --env staging

# Production 배포
uv run python scripts/deploy_source.py --env production

# 버전 등록
uv run python scripts/version/register.py \
    --version v1.0.0 \
    --env production
```

## Project Structure

```
AgentEngine/
├── pyproject.toml              # 패키지 설정
├── cloudbuild.yaml             # CI/CD 파이프라인
├── src/
│   └── agent_engine/
│       ├── __init__.py
│       ├── agent.py            # Agent Wrapper (핵심)
│       ├── config.py           # 설정 관리
│       ├── exceptions.py       # 예외 정의
│       ├── tools/              # 기본 Tools
│       │   ├── search.py       # 검색 (Mock)
│       │   ├── calculator.py   # 계산
│       │   ├── datetime_tool.py # 날짜/시간
│       │   └── memory_tools.py # 메모리 저장/조회
│       ├── sessions/           # Session 관리
│       │   ├── manager.py
│       │   ├── models.py
│       │   └── backends/       # InMemory, VertexAI
│       ├── memory/             # Memory Bank
│       │   ├── manager.py
│       │   ├── retriever.py
│       │   └── backends/       # InMemory, VertexAI
│       ├── observability/      # Tracing, Logging, Metrics
│       ├── evaluation/         # Agent 평가
│       └── version/            # 버전 관리
├── scripts/
│   ├── deploy.py               # SDK 배포
│   ├── deploy_source.py        # 소스 배포
│   └── version/                # 버전 관리 스크립트
├── monitoring/
│   ├── dashboard.yaml          # 모니터링 대시보드
│   └── alerts.yaml             # 알림 설정
├── docs/
│   ├── API_REFERENCE.md        # API 문서
│   ├── DEPLOYMENT.md           # 배포 가이드
│   ├── SECURITY.md             # 보안 가이드
│   ├── RUNBOOK.md              # 운영 런북
│   ├── TRAINING.md             # 교육 자료
│   └── OPERATIONS.md           # 운영 가이드
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
└── tests/
    ├── test_agent.py
    ├── test_tools.py
    ├── test_sessions.py
    ├── test_memory.py
    ├── test_observability.py
    └── evaluation/
```

## Configuration

### 환경 변수

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `AGENT_PROJECT_ID` | GCP 프로젝트 ID | - |
| `AGENT_LOCATION` | GCP 리전 | `asia-northeast3` |
| `AGENT_MODEL` | Gemini 모델 | `gemini-2.5-pro` |
| `AGENT_TEMPERATURE` | 모델 온도 | `0.7` |
| `AGENT_MAX_TOKENS` | 최대 토큰 | `4096` |
| `AGENT_SYSTEM_PROMPT` | 시스템 프롬프트 | `You are a helpful AI assistant.` |
| `SESSION_ENABLED` | 세션 활성화 | `false` |
| `SESSION_TTL_HOURS` | 세션 TTL | `24` |
| `MEMORY_ENABLED` | 메모리 활성화 | `false` |
| `MEMORY_AUTO_GENERATE` | 메모리 자동 생성 | `false` |

### AgentConfig

```python
from agent_engine import (
    AgentConfig,
    SessionConfig,
    SessionBackendType,
    MemoryConfig,
    MemoryBackendType,
    ObservabilityConfig,
)

config = AgentConfig(
    model="gemini-2.5-pro",
    project_id="your-project-id",
    location="asia-northeast3",
    system_prompt="You are a helpful assistant.",
    temperature=0.7,
    max_tokens=4096,

    # Session 설정
    session=SessionConfig(
        enabled=True,
        backend_type=SessionBackendType.IN_MEMORY,  # or VERTEX_AI
        ttl_hours=24,
    ),

    # Memory 설정
    memory=MemoryConfig(
        enabled=True,
        backend_type=MemoryBackendType.IN_MEMORY,  # or VERTEX_AI
        auto_generate=True,
        max_memories_per_user=100,
    ),

    # Observability 설정
    observability=ObservabilityConfig(
        tracing_enabled=True,
        logging_enabled=True,
        metrics_enabled=True,
    ),
)
```

## Docker

```bash
# 개발 환경 시작
docker-compose -f docker/docker-compose.yml up agent

# 테스트 실행
docker-compose -f docker/docker-compose.yml --profile test up test

# 린팅
docker-compose -f docker/docker-compose.yml --profile lint up lint
```

## API Reference

자세한 API 문서는 [docs/API_REFERENCE.md](docs/API_REFERENCE.md)를 참조하세요.

### 주요 클래스

| 클래스 | 설명 |
|--------|------|
| `PydanticAIAgentWrapper` | Agent Engine 규격 준수 Agent Wrapper |
| `StreamChunk` | 스트리밍 응답 청크 |
| `AgentConfig` | Agent 설정 |
| `SessionConfig` | 세션 설정 |
| `MemoryConfig` | 메모리 설정 |

### 주요 메서드

| 메서드 | 설명 |
|--------|------|
| `query()` | 동기 쿼리 |
| `aquery()` | 비동기 쿼리 |
| `stream_query()` | 비동기 스트리밍 쿼리 |
| `stream_query_sync()` | 동기 스트리밍 쿼리 |
| `query_with_session()` | 세션 기반 비동기 쿼리 |
| `stream_query_with_session()` | 세션 기반 스트리밍 쿼리 |

## Documentation

- [API Reference](docs/API_REFERENCE.md) - 전체 API 문서
- [Deployment Guide](docs/DEPLOYMENT.md) - 배포 가이드
- [Security Guide](docs/SECURITY.md) - 보안 가이드
- [Operations Runbook](docs/RUNBOOK.md) - 운영 런북
- [Training Guide](docs/TRAINING.md) - 팀 교육 자료
- [Operations Guide](docs/OPERATIONS.md) - 운영 가이드

## Project Status

### Phase 1: Foundation (Week 1-4) ✅
- Agent Engine 규격 준수, Pydantic AI 통합, 기본 Tool

### Phase 2: Core Features (Week 5-8) ✅
- Session 관리, Memory Bank, Observability

### Phase 3: Production Readiness (Week 9-12) ✅
- CI/CD, 버전 관리, 평가 자동화, 보안

### Phase 4: Advanced Features (Week 13-16) ✅
- 스트리밍 응답, 문서화 완료

## License

MIT
