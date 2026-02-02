# Pydantic AI Agent Platform on VertexAI Agent Engine

VertexAI Agent Engine에 배포 가능한 Pydantic AI 기반 Agent 플랫폼입니다.

## Features

- **Agent Engine 규격 준수**: `__init__`, `set_up`, `query` 메서드 구현
- **Pydantic AI 통합**: Pydantic AI Agent를 Agent Engine에서 실행
- **Gemini 모델 연동**: GoogleProvider를 통한 VertexAI Gemini 모델 사용
- **동기/비동기 지원**: `query` (sync), `aquery` (async) 메서드 제공
- **기본 Tool 라이브러리**: 검색, 계산, 날짜/시간 Tool 포함
- **Docker 지원**: 로컬 개발/테스트 환경 제공

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

### 2. 로컬 테스트

```bash
# 테스트 실행
uv run pytest tests/ -v

# 린팅
uv run ruff check src/ tests/

# 타입 체크
uv run mypy src/
```

### 3. Agent Engine 배포

```bash
# SDK 기반 배포
uv run python scripts/deploy.py deploy \
    --project YOUR_PROJECT_ID \
    --location asia-northeast3

# 소스 기반 배포 (CI/CD용)
uv run python scripts/deploy_source.py deploy \
    --project YOUR_PROJECT_ID \
    --location asia-northeast3
```

## Project Structure

```
AgentEngine/
├── pyproject.toml              # 패키지 설정
├── src/
│   └── agent_engine/
│       ├── __init__.py
│       ├── agent.py            # Agent Wrapper (핵심)
│       ├── config.py           # 설정 관리
│       ├── exceptions.py       # 예외 정의
│       └── tools/              # 기본 Tools
│           ├── search.py       # 검색 (Mock)
│           ├── calculator.py   # 계산
│           └── datetime_tool.py # 날짜/시간
├── scripts/
│   ├── deploy.py               # SDK 배포
│   └── deploy_source.py        # 소스 배포
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
└── tests/
    ├── test_agent.py
    └── test_tools.py
```

## Usage Example

```python
from agent_engine import PydanticAIAgentWrapper
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

# 쿼리 실행
response = agent.query(
    message="What time is it in Seoul?",
    user_id="user123",
)

print(response["response"])
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

## Configuration

환경 변수로 설정:

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `AGENT_PROJECT_ID` | GCP 프로젝트 ID | - |
| `AGENT_LOCATION` | GCP 리전 | `asia-northeast3` |
| `AGENT_MODEL` | Gemini 모델 | `gemini-2.5-pro` |
| `AGENT_TEMPERATURE` | 모델 온도 | `0.7` |
| `AGENT_MAX_TOKENS` | 최대 토큰 | `4096` |
| `AGENT_SYSTEM_PROMPT` | 시스템 프롬프트 | `You are a helpful AI assistant.` |

## Phase 1 Checklist

- [x] Python 3.11+ 개발 환경 구성
- [x] uv 패키지 매니저 설정
- [x] pyproject.toml 작성
- [x] Pydantic AI 의존성 추가
- [x] Agent Engine 규격 준수 (AC-001)
- [x] Pydantic AI Agent 래핑 (AC-002)
- [x] GoogleProvider Gemini 연동 (AC-003)
- [x] 동기/비동기 지원 (AC-004)
- [x] 기본 에러 핸들링 (AC-006)
- [x] Tool 등록 메커니즘 (TS-001)
- [x] 기본 Tool 구현 (TS-002)
- [x] SDK 배포 스크립트 (DM-001)
- [x] 소스 배포 스크립트 (DM-002)
- [x] Docker 환경 구축
- [x] 단위 테스트 작성

## Next Steps (Phase 2)

- Session 관리 통합
- Memory Bank 통합
- Cloud Trace/Logging 연동
- 모니터링 대시보드

## License

MIT
