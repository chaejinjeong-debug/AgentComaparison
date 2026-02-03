# TODO: Pydantic AI Agent Platform on VertexAI Agent Engine

**최종 수정일**: 2026-02-03
**상태**: Phase 4 In Progress (Architecture Refactoring Complete)

---

## Phase 1: Foundation (Week 1-4) ✅

### Week 1: 개발 환경 설정
- [x] Python 3.11+ 개발 환경 구성
- [x] uv 패키지 매니저 설치 및 설정
- [x] `pyproject.toml` 작성 (uv 기반)
- [x] Pydantic AI (>= 1.51.0) 의존성 추가
- [x] GCP 프로젝트 설정 및 인증 구성
- [x] VertexAI Agent Engine 연동 테스트

### Week 2: Agent Wrapper 개발
- [x] **AC-001**: Agent Engine 규격 준수 (`__init__`, `set_up`, `query` 메서드)
- [x] **AC-002**: Pydantic AI Agent 래핑 구현
- [x] **AC-003**: GoogleProvider를 통한 Gemini 모델 연동
- [x] **AC-004**: 동기/비동기 지원 (`query`, `aquery`)
- [x] **AC-006**: 기본 에러 핸들링 구현
- [x] **TS-001**: Tool 등록 메커니즘 구현 (@tool 데코레이터)
- [x] **TS-002**: 기본 Tool 구현 (검색, 계산, 날짜/시간)

### Week 3: 배포 파이프라인
- [x] **DM-001**: SDK 기반 배포 스크립트 개발
- [x] Agent Engine 배포 테스트
- [x] Docker 기반 로컬 개발/테스트 환경 구축
  - [x] `Dockerfile` 작성
  - [x] `docker-compose.yml` 작성
  - [x] 로컬 Agent Engine 에뮬레이션 환경 구성

### Week 4: 소스 배포 및 테스트
- [x] **DM-002**: 소스 기반 배포 구현 (CI/CD용)
- [x] **PM-001**: System Prompt 설정 (환경 변수/설정 파일)
- [x] 단위 테스트 작성
- [x] Phase 1 문서화

**Milestone**: Agent Engine 배포 및 기본 쿼리 동작 확인 ✅

---

## Phase 2: Core Features (Week 5-8) ✅

### Week 5: Session 관리 통합
- [x] **SM-001**: 세션 생성 기능 구현
- [x] **SM-002**: 세션 이벤트 기록 (AppendEvent)
- [x] **SM-003**: 세션 이력 조회 (ListEvents)
- [x] **SM-004**: 세션 TTL 관리 (기본 24시간)
- [x] **SM-005**: 세션 삭제 기능

### Week 6: Memory Bank 통합
- [x] **MB-001**: 메모리 자동 생성 (fact 추출)
- [x] **MB-002**: 사용자별 메모리 조회
- [x] **MB-003**: Similarity Search 구현
- [x] **MB-004**: Agent 명시적 메모리 저장
- [x] **MB-006**: 메모리 삭제 기능 (GDPR 대응)

### Week 7: Observability 연동
- [x] **OB-001**: Cloud Trace 통합
- [x] **OB-002**: Cloud Logging 구조화
- [x] **OB-003**: 메트릭 수집 (요청 수, 에러율, 토큰 사용량)
- [x] **TS-004**: Tool 실행 로깅

### Week 8: 모니터링 대시보드
- [x] **OB-004**: 모니터링 대시보드 구축
- [x] 통합 테스트 작성
- [x] Phase 2 문서화

**Milestone**: Session, Memory, Observability 완전 동작 ✅

---

## Phase 2.5: Backend Abstraction ✅

### Session/Memory 영속성 리팩토링
- [x] Strategy Pattern 적용하여 Backend 추상화
- [x] SessionBackend ABC 및 구현체 (InMemory, VertexAI)
- [x] MemoryBackend ABC 및 구현체 (InMemory, VertexAI)
- [x] Config에 SessionBackendType, MemoryBackendType enum 추가
- [x] 기존 테스트 128개 통과 확인

### SDK 마이그레이션 (REST API → vertexai.Client())
- [x] Sessions Backend: httpx REST API → vertexai.Client() SDK
- [x] Memory Backend: httpx REST API → vertexai.Client() SDK
- [x] `_get_access_token()`, `_get_headers()` 제거 (SDK가 인증 처리)
- [x] pyproject.toml에서 httpx 의존성 제거
- [x] 테스트 128개 통과 확인

**SDK 사용 예시:**
```python
import vertexai
client = vertexai.Client(project=project_id, location=location)

# Sessions
session = client.agent_engines.sessions.create(name=agent_engine_name, user_id=user_id)
client.agent_engines.sessions.events.append(name=session_name, author="user", ...)
client.agent_engines.sessions.delete(name=session_name)

# Memory Bank
memory = client.agent_engines.memories.create(name=agent_engine_name, fact=fact, scope=scope)
client.agent_engines.memories.retrieve(name=agent_engine_name, scope=scope)
client.agent_engines.memories.delete(name=memory_name)
client.agent_engines.memories.purge(name=agent_engine_name, filter=filter_string, force=True)
```

**변경 파일:**
- `config.py`: Backend 타입 enum 추가
- `sessions/backends/`: 신규 디렉토리 (base.py, in_memory.py, vertex_ai.py)
- `sessions/manager.py`: Backend 위임 패턴 적용
- `memory/backends/`: 신규 디렉토리 (base.py, in_memory.py, vertex_ai.py)
- `memory/manager.py`: Backend 위임 패턴 적용
- `pyproject.toml`: httpx 의존성 제거

---

## Phase 2 구현 상세

### 신규 모듈
```
src/agent_engine/
├── sessions/                 # Session Management (SM-001~005)
│   ├── __init__.py
│   ├── manager.py           # SessionManager (Backend 위임)
│   ├── models.py            # Session, SessionEvent models
│   └── backends/            # Storage Backends
│       ├── __init__.py
│       ├── base.py          # SessionBackend ABC
│       ├── in_memory.py     # InMemorySessionBackend
│       └── vertex_ai.py     # VertexAISessionBackend
├── memory/                   # Memory Bank (MB-001~006)
│   ├── __init__.py
│   ├── manager.py           # MemoryManager (Backend 위임)
│   ├── retriever.py         # Similarity Search
│   ├── models.py            # Memory, MemoryScope models
│   └── backends/            # Storage Backends
│       ├── __init__.py
│       ├── base.py          # MemoryBackend ABC
│       ├── in_memory.py     # InMemoryMemoryBackend
│       └── vertex_ai.py     # VertexAIMemoryBackend
├── observability/            # Observability (OB-001~003, TS-004)
│   ├── __init__.py
│   ├── tracing.py           # Cloud Trace
│   ├── logging.py           # Structured Logging
│   ├── metrics.py           # Metrics Collection
│   └── decorators.py        # @traced, @metered, @logged_tool
└── tools/
    └── memory_tools.py      # save_memory, recall_user_info tools

monitoring/
└── dashboard.yaml           # Cloud Monitoring Dashboard (OB-004)

tests/
├── test_sessions.py         # Session Management tests
├── test_memory.py           # Memory Bank tests
├── test_observability.py    # Observability tests
└── test_integration.py      # Integration tests
```

### 주요 변경 파일
- `config.py`: SessionConfig, MemoryConfig, ObservabilityConfig 추가
- `agent.py`: query_with_session(), 메모리/세션 통합, 메트릭 기록
- `__init__.py`: Phase 2 모듈 export
- `pyproject.toml`: OpenTelemetry, GCP 모니터링 의존성 추가
- `conftest.py`: Phase 2 테스트 fixtures 추가

---

## Phase 3: Production Readiness (Week 9-12) ✅

### Week 9: CI/CD 파이프라인
- [x] Cloud Build 기반 CI/CD 구축
- [x] uv 기반 의존성 설치 자동화
- [x] 자동화된 테스트 실행 (pytest)
- [x] 코드 품질 게이트 (ruff, mypy, bandit)
- [x] Coverage report 생성

### Week 10: 버전 관리 시스템
- [x] **DM-006**: 버전 관리 시스템 구현
- [x] **DM-007**: 롤백 메커니즘 구현
- [x] **DM-008**: 환경 분리 (Staging/Production)
- [x] 버전 레지스트리 관리

### Week 11: Agent 평가 자동화
- [x] Agent 품질 평가 (Evaluation) 구현
- [x] 응답 정확도 검증
- [x] 품질 임계치 설정 (85%)
- [x] 성능 테스트 (P50 < 2s, P99 < 10s)

### Week 12: 보안 및 Production 배포
- [x] VPC-SC 설정
- [x] IAM 최소 권한 원칙 적용
- [x] CMEK 설정 (필요시)
- [x] **OB-005**: 알림 정책 구성
- [x] Production 배포

**Milestone**: Production 환경 운영 시작 ✅

---

## Phase 4: Advanced Features (Week 13-16) ✅

### Week 13: 스트리밍 응답
- [x] **AC-005**: `stream_query` 메서드 구현
- [x] Bidirectional streaming 지원

### Week 13.5: 아키텍처 리팩토링 ✅
- [x] Core 모듈 분리 (`agent_engine/core/`)
- [x] **BaseAgent**: 추상 인터페이스 정의
- [x] **PydanticAIAgent**: Pydantic AI 전용 비즈니스 로직 분리
- [x] **AgentEngineWrapper**: 배포/오케스트레이션 로직 분리
- [x] **MessageBuilder**: 메시지 조합 유틸리티
- [x] **ResultProcessor**: 결과 처리 유틸리티
- [x] `PydanticAIAgentWrapper` Facade 패턴으로 하위 호환성 유지
- [x] 신규 52개 테스트 추가 (test_core.py)
- [x] 기존 테스트 27개 통과 확인

**리팩토링 결과:**
- `agent.py`: 1143줄 → 1079줄 (Facade)
- 각 core 클래스: ~100-300줄
- 향후 다른 LLM Provider 지원 가능 구조

### Week 14: 멀티 에이전트
- [ ] A2A Protocol 설계
- [ ] Agent 간 통신 구현
- [ ] 멀티 에이전트 오케스트레이션

### Week 15: 고급 기능
- [ ] **TS-003**: 커스텀 Tool 확장 가이드
- [ ] **TS-005**: Tool 타임아웃 설정
- [ ] **PM-002**: Dynamic Prompt 구현
- [ ] **SM-006**: 세션 메타데이터 지원
- [ ] **MB-005**: 메모리 Revision (버전 관리)
- [ ] 비용 최적화

### Week 16: 마무리
- [x] 전체 문서화 완료
- [x] 팀 교육 자료 작성
- [x] 운영 가이드 작성
- [x] 프로젝트 회고

**Milestone**: 전체 기능 완료 및 안정화

---

## 비기능 요구사항 체크리스트

### 성능
- [ ] P50 응답 지연 시간 < 2초
- [ ] P99 응답 지연 시간 < 10초
- [ ] 처리량 > 100 QPS
- [ ] Cold Start < 30초

### 가용성
- [ ] SLA 99.9% 달성
- [ ] RTO < 5분
- [ ] RPO = 0 (데이터 손실 없음)

### 확장성
- [ ] Auto-scaling 설정 (min: 1, max: 100)
- [ ] CPU 70% 기준 스케일링

### 보안
- [ ] VPC-SC 적용
- [ ] IAM 최소 권한 적용
- [ ] 감사 로그 활성화
- [ ] PII 마스킹 처리

---

## 기술 스택

| 영역 | 기술 | 비고 |
|------|------|------|
| **Package Manager** | uv | 빠른 의존성 관리 |
| **Local Dev** | Docker + docker-compose | 로컬 테스트 환경 |
| **Runtime** | VertexAI Agent Engine | 프로덕션 배포 |
| **Framework** | Pydantic AI >= 1.51.0 | Agent 프레임워크 |
| **LLM** | Gemini 2.5 Pro/Flash | VertexAI API |
| **Language** | Python 3.11+ | |
| **Test** | pytest | 단위/통합 테스트 |
| **Lint** | ruff, mypy | 코드 품질 |
| **CI/CD** | Cloud Build | GCP 네이티브 |
| **Observability** | OpenTelemetry | Cloud Trace/Monitoring |

---

## 참고 사항

- 각 요구사항 ID는 PRD.md 참조 (예: AC-001, SM-001, MB-001 등)
- 우선순위: P0 (필수) > P1 (중요) > P2 (권장)
- Session/Memory Bank 과금 시작: 2026.01.28~ (비용 모니터링 필요)
