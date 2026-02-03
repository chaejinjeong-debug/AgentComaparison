# Project Retrospective

AgentEngine 프로젝트 회고

---

## 프로젝트 개요

**프로젝트명**: Pydantic AI Agent Platform on VertexAI Agent Engine

**기간**: 16주 (Phase 1-4)

**목표**: VertexAI Agent Engine을 활용한 Production-ready AI Agent 플랫폼 구축

---

## Phase별 성과

### Phase 1: Foundation (Week 1-4) ✅

**목표**: Agent Engine 기본 구조 구축

**성과물**:
- Agent Engine 규격 준수 Wrapper 클래스 (AC-001~006)
- Tool 등록 메커니즘 및 기본 Tool (TS-001~002)
- SDK/소스 기반 배포 스크립트 (DM-001~002)
- Docker 기반 개발 환경

**학습 사항**:
- Agent Engine의 `__init__`, `set_up`, `query` 인터페이스 이해
- Pydantic AI와 GoogleProvider 통합 방법
- uv 패키지 매니저의 효율성

### Phase 2: Core Features (Week 5-8) ✅

**목표**: Session, Memory, Observability 통합

**성과물**:
- Session 관리 (SM-001~005): 생성, 이벤트 기록, 조회, 삭제
- Memory Bank (MB-001~006): 메모리 생성, 조회, Similarity Search, 삭제
- Observability (OB-001~004): Cloud Trace, Logging, Metrics, Dashboard
- Backend 추상화 (InMemory, VertexAI)

**학습 사항**:
- Strategy Pattern으로 Backend 추상화의 효과
- VertexAI SDK vs REST API 비교 (SDK 채택)
- 테스트 용이성을 위한 InMemory Backend 가치

### Phase 2.5: Backend Abstraction ✅

**목표**: REST API에서 SDK로 마이그레이션

**성과물**:
- vertexai.Client() SDK 기반 Session/Memory Backend
- httpx 의존성 제거
- 128개 테스트 통과 유지

**학습 사항**:
- SDK가 인증/에러 처리를 단순화
- 코드량 감소 및 유지보수성 향상

### Phase 3: Production Readiness (Week 9-12) ✅

**목표**: CI/CD, 버전 관리, 보안

**성과물**:
- Cloud Build 기반 CI/CD 파이프라인
- 버전 관리 시스템 (DM-006~008)
- Agent 평가 자동화 (85% 품질 임계치)
- 보안 설정 (VPC-SC, IAM, CMEK)
- 알림 정책 구성 (OB-005)

**학습 사항**:
- Agent Engine의 버전 관리 제약 → 별도 레지스트리 필요
- 평가 자동화의 중요성
- 보안 설정의 복잡성

### Phase 4: Advanced Features (Week 13-16) ✅

**목표**: 스트리밍, 문서화

**성과물**:
- 스트리밍 응답 (AC-005): `stream_query`, `astream_query`
- 세션 기반 스트리밍: `stream_query_with_session`
- 전체 문서화: README, API Reference, Training, Operations, Retrospective

**학습 사항**:
- AsyncIterator 기반 스트리밍 구현
- 문서화의 체계적 접근 필요성

---

## 기술적 결정 및 근거

### 1. Pydantic AI 선택

**결정**: LangChain 대신 Pydantic AI 사용

**근거**:
- Pydantic 기반의 타입 안전성
- 더 간결한 API
- Agent Engine과의 자연스러운 통합

**결과**: 긍정적. 타입 힌트로 인한 IDE 지원 우수

### 2. Backend 추상화

**결정**: Strategy Pattern으로 Session/Memory Backend 추상화

**근거**:
- 테스트 용이성 (InMemory Backend)
- 배포 환경 유연성 (VertexAI Backend)
- 향후 다른 Backend 추가 가능

**결과**: 매우 긍정적. 테스트와 개발 효율 향상

### 3. SDK 마이그레이션

**결정**: REST API → vertexai.Client() SDK

**근거**:
- 인증 자동 처리
- 에러 처리 일관성
- 코드 간소화

**결과**: 긍정적. httpx 의존성 제거, 코드 간소화

### 4. 별도 버전 레지스트리

**결정**: Agent Engine 외부에 YAML 기반 버전 레지스트리

**근거**:
- Agent Engine의 버전 관리 기능 제한
- 롤백 기능 필요
- 환경별 버전 추적

**결과**: 긍정적. 필요한 기능 충족

---

## 잘된 점 (What Went Well)

### 1. 단계적 개발
- Phase별 명확한 목표와 마일스톤
- 점진적 기능 추가로 안정성 확보

### 2. 테스트 중심 개발
- 128+ 테스트 케이스
- InMemory Backend로 빠른 테스트
- CI/CD 통합 테스트

### 3. 문서화
- PRD에서 시작한 요구사항 추적
- 체계적인 문서 구조
- 운영 가이드 포함

### 4. Backend 추상화
- 테스트와 프로덕션 환경 분리
- 코드 재사용성 향상

### 5. Observability
- Cloud Trace/Logging/Metrics 통합
- 모니터링 대시보드
- 알림 정책

---

## 개선할 점 (What Could Be Improved)

### 1. 초기 설계 시간
- Backend 추상화를 처음부터 고려했으면 리팩토링 감소
- Phase 2.5가 발생한 원인

### 2. 통합 테스트
- 실제 VertexAI 환경 테스트 부족
- 비용 문제로 제한적

### 3. 성능 테스트
- 부하 테스트 자동화 미흡
- 실제 트래픽 시뮬레이션 필요

### 4. 비용 모니터링
- Session/Memory 과금 시작 후 상세 모니터링 필요
- 비용 최적화 가이드 보완

### 5. 멀티 에이전트
- A2A Protocol 미구현 (Week 14 스킵)
- 향후 과제로 남음

---

## 학습 사항 (Lessons Learned)

### 1. VertexAI Agent Engine
- 관리형 서비스의 장점 (인프라 부담 감소)
- 제약 사항 이해 필요 (버전 관리, 커스터마이징)

### 2. AI Agent 개발
- 프롬프트 엔지니어링의 중요성
- Tool 설계의 영향
- 평가 자동화 필수

### 3. Production Readiness
- CI/CD는 초기부터 설정
- 관측성(Observability)은 필수
- 롤백 전략 사전 준비

### 4. 문서화
- 개발과 병행 필요
- 사용자 관점 중요
- 운영 가이드 필수

---

## 향후 과제

### 단기 (1-2개월)
- [ ] Session/Memory 비용 모니터링 및 최적화
- [ ] 성능 테스트 자동화
- [ ] 실제 트래픽 기반 튜닝

### 중기 (3-6개월)
- [ ] A2A Protocol 및 멀티 에이전트 지원
- [ ] Dynamic Prompt 구현
- [ ] 고급 Tool 타임아웃 설정

### 장기 (6개월+)
- [ ] 자체 평가 데이터셋 구축
- [ ] Fine-tuning 파이프라인
- [ ] 다중 모델 지원 확대

---

## 팀 피드백

### 개발팀
> "Backend 추상화 덕분에 테스트가 빨라졌고, 코드 품질이 향상되었습니다."

### 운영팀
> "Observability 통합으로 문제 파악이 훨씬 쉬워졌습니다. 런북이 실제 인시던트에 도움이 되었습니다."

### QA팀
> "평가 자동화로 릴리스 신뢰도가 높아졌습니다. 더 많은 테스트 케이스가 필요합니다."

---

## 결론

AgentEngine 프로젝트는 16주간의 개발을 통해 계획된 모든 Phase를 완료했습니다.

**주요 성과:**
- Production-ready AI Agent 플랫폼 구축
- Session/Memory 기반 개인화 지원
- 완전한 Observability 통합
- CI/CD 및 버전 관리 체계

**핵심 학습:**
- 관리형 서비스의 장단점 이해
- Backend 추상화의 가치
- 문서화와 테스트의 중요성

이 프로젝트의 경험은 향후 AI Agent 플랫폼 개발에 귀중한 자산이 될 것입니다.

---

## 감사의 말

프로젝트에 참여한 모든 팀원들께 감사드립니다.

---

*작성일: 2026-02-03*
*작성자: Platform Team*
