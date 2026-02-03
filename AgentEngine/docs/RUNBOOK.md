# Operations Runbook

AgentEngine 운영 런북

## Overview

이 문서는 AgentEngine 운영 중 발생할 수 있는 문제와 대응 방법을 설명합니다.

---

## 알림 대응 절차

### high-error-rate

**알림**: Agent Engine - High Error Rate

**심각도**: WARNING

**증상**: 에러율이 5%를 초과

#### 조사 단계

1. **로그 확인**
   ```bash
   gcloud logging read \
     'resource.type="aiplatform.googleapis.com/AgentEngine" severity>=ERROR' \
     --limit=50 \
     --project=YOUR_PROJECT_ID \
     --format=json
   ```

2. **에러 패턴 분석**
   ```bash
   # 에러 유형별 카운트
   gcloud logging read \
     'severity>=ERROR' \
     --format='value(jsonPayload.error_type)' | sort | uniq -c | sort -rn
   ```

3. **최근 배포 확인**
   ```bash
   python scripts/version/rollback.py --env production --list
   ```

#### 대응 조치

1. **일시적 에러인 경우**: 모니터링 계속
2. **특정 요청 유형에서 발생**: 해당 기능 조사
3. **전체적 에러 증가**: 롤백 고려
   ```bash
   python scripts/version/rollback.py \
     --env production \
     --reason "High error rate after deployment"
   ```

---

### high-latency

**알림**: Agent Engine - High P99 Latency

**심각도**: WARNING

**증상**: P99 지연시간이 10초 초과

#### 조사 단계

1. **Cloud Trace 확인**
   ```bash
   # Trace 목록 조회
   gcloud trace traces list \
     --project=YOUR_PROJECT_ID \
     --filter='latency>10s'
   ```

2. **느린 요청 패턴 분석**
   - 특정 프롬프트/쿼리 유형
   - 특정 시간대
   - 특정 사용자

3. **리소스 사용량 확인**
   ```bash
   # Vertex AI 할당량 확인
   gcloud compute project-info describe \
     --project=YOUR_PROJECT_ID
   ```

#### 대응 조치

1. **프롬프트 복잡도 문제**: 프롬프트 최적화
2. **모델 지연**: 더 빠른 모델로 전환 (Flash)
3. **리소스 부족**: 스케일링 설정 조정
4. **외부 API 지연**: 타임아웃 설정 확인

---

### service-down

**알림**: Agent Engine - Service Down

**심각도**: CRITICAL

**증상**: 5분간 요청 없음

#### 즉시 조치

1. **상태 확인**
   ```bash
   # Agent Engine 상태
   gcloud ai agent-engines describe AGENT_ENGINE_ID \
     --project=YOUR_PROJECT_ID

   # 최근 배포 상태
   gcloud ai agent-engines operations list \
     --project=YOUR_PROJECT_ID \
     --limit=5
   ```

2. **네트워크 확인**
   ```bash
   # VPC-SC 위반 확인
   gcloud logging read \
     'protoPayload.status.code="7"' \
     --limit=10
   ```

3. **헬스체크 수동 실행**
   ```bash
   curl -X GET https://YOUR_ENDPOINT/health
   ```

#### 복구 절차

1. **배포 문제인 경우**
   ```bash
   # 롤백
   python scripts/version/rollback.py \
     --env production \
     --reason "Service down"
   ```

2. **인프라 문제인 경우**
   - GCP 상태 페이지 확인
   - 지원 케이스 생성

3. **네트워크 문제인 경우**
   - VPC-SC 설정 확인
   - 방화벽 규칙 확인

---

### memory-bank-errors

**알림**: Agent Engine - Memory Bank Errors

**심각도**: WARNING

**증상**: Memory Bank 작업 에러율 10% 초과

#### 조사 단계

1. **Memory Bank 상태 확인**
   ```bash
   gcloud logging read \
     'jsonPayload.component="memory_bank"' \
     --limit=50
   ```

2. **할당량 확인**
   - Memory Bank 저장 용량
   - 초당 요청 수

#### 대응 조치

1. **용량 문제**: 오래된 메모리 정리
2. **API 문제**: 재시도 로직 확인
3. **데이터 문제**: 데이터 검증

---

### session-failures

**알림**: Agent Engine - Session Failures

**심각도**: WARNING

**증상**: 세션 작업 에러율 5% 초과

#### 조사 단계

1. **세션 로그 확인**
   ```bash
   gcloud logging read \
     'jsonPayload.component="sessions"' \
     --limit=50
   ```

2. **세션 만료 확인**
   - TTL 설정 확인
   - 만료된 세션 접근 시도

#### 대응 조치

1. **TTL 문제**: 설정 조정
2. **동시성 문제**: 락/동기화 확인

---

### high-token-usage

**알림**: Agent Engine - High Token Usage

**심각도**: WARNING

**증상**: 시간당 100만 토큰 초과

#### 조사 단계

1. **토큰 사용량 분석**
   ```bash
   # 사용자별 토큰 사용량
   gcloud logging read \
     'jsonPayload.token_usage>0' \
     --format='value(jsonPayload.user_id,jsonPayload.token_usage)'
   ```

2. **비정상 패턴 확인**
   - 특정 사용자의 과다 사용
   - 반복 요청
   - 대용량 프롬프트

#### 대응 조치

1. **비용 관리**: 사용량 제한 설정
2. **최적화**: 프롬프트 길이 최적화
3. **캐싱**: 반복 요청 캐싱

---

### security-incident

**알림**: Agent Engine - Security: Unauthorized Access

**심각도**: CRITICAL

**증상**: 분당 10회 이상 권한 거부

#### 즉시 조치

1. **감사 로그 확인**
   ```bash
   gcloud logging read \
     'protoPayload.authenticationInfo.principalEmail!=""
      protoPayload.status.code="7"' \
     --limit=100 \
     --format=json
   ```

2. **의심스러운 IP/계정 식별**
   ```bash
   # IP별 접근 시도
   gcloud logging read \
     'protoPayload.status.code="7"' \
     --format='value(protoPayload.requestMetadata.callerIp)' | sort | uniq -c
   ```

3. **보안팀 알림**
   - 인시던트 티켓 생성
   - 보안 담당자 호출

#### 대응 조치

1. **계정 침해 의심**: 계정 비활성화
2. **API 키 노출**: 키 로테이션
3. **IP 기반 공격**: 방화벽 규칙 추가

---

## 정기 운영 작업

### 일일 점검

- [ ] 대시보드 확인
- [ ] 에러 로그 검토
- [ ] 알림 확인

### 주간 점검

- [ ] 성능 지표 리뷰
- [ ] 비용 분석
- [ ] 보안 로그 검토

### 월간 점검

- [ ] IAM 권한 검토
- [ ] 버전 정리
- [ ] 문서 업데이트

---

## 유용한 명령어

### 로그 조회

```bash
# 최근 에러
gcloud logging read 'severity>=ERROR' --limit=10

# 특정 시간대
gcloud logging read 'timestamp>="2024-01-01T00:00:00Z"' --limit=100

# JSON 형식
gcloud logging read 'severity>=ERROR' --format=json
```

### 메트릭 조회

```bash
# 알림 정책 목록
gcloud alpha monitoring policies list

# 알림 상태
gcloud alpha monitoring policies describe POLICY_ID
```

### 리소스 관리

```bash
# Agent Engine 목록
gcloud ai agent-engines list

# Agent Engine 삭제
gcloud ai agent-engines delete AGENT_ENGINE_ID
```

---

## 에스컬레이션

| 레벨 | 담당 | 연락처 |
|------|------|--------|
| L1 | 온콜 엔지니어 | oncall@your-domain.com |
| L2 | 플랫폼 팀 | platform@your-domain.com |
| L3 | GCP 지원 | support.google.com |

---

## 관련 문서

- [SECURITY.md](./SECURITY.md) - 보안 가이드
- [DEPLOYMENT.md](./DEPLOYMENT.md) - 배포 가이드
- [monitoring/alerts.yaml](../monitoring/alerts.yaml) - 알림 설정
