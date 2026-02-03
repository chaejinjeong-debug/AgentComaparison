# Operations Guide

AgentEngine 운영 가이드

---

## 1. 운영 개요

### 1.1 서비스 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Production Environment                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   ┌─────────────┐        ┌─────────────────────────────────────┐    │
│   │   Clients   │───────►│        Load Balancer (HTTPS)        │    │
│   └─────────────┘        └───────────────┬─────────────────────┘    │
│                                          │                           │
│                          ┌───────────────▼─────────────────┐        │
│                          │     VertexAI Agent Engine        │        │
│                          │                                  │        │
│                          │  ┌────────────────────────────┐ │        │
│                          │  │    AgentEngine Instance    │ │        │
│                          │  │                            │ │        │
│                          │  │  ┌──────┐ ┌──────┐       │ │        │
│                          │  │  │Agent │ │Tools │       │ │        │
│                          │  │  └──────┘ └──────┘       │ │        │
│                          │  └────────────────────────────┘ │        │
│                          │                                  │        │
│                          │  ┌─────────────┬──────────────┐ │        │
│                          │  │  Sessions   │  Memory Bank │ │        │
│                          │  └─────────────┴──────────────┘ │        │
│                          └──────────────────────────────────┘        │
│                                          │                           │
│                          ┌───────────────▼─────────────────┐        │
│                          │        Gemini API (VertexAI)     │        │
│                          └─────────────────────────────────┘        │
│                                          │                           │
│         ┌────────────────────────────────┼────────────────────────┐  │
│         │                                │                        │  │
│   ┌─────▼─────┐    ┌───────────────┐    │    ┌─────────────────┐ │  │
│   │ Cloud     │    │ Cloud         │    │    │ Cloud           │ │  │
│   │ Trace     │    │ Logging       │    │    │ Monitoring      │ │  │
│   └───────────┘    └───────────────┘    │    └─────────────────┘ │  │
│                                          │                        │  │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 서비스 레벨 목표 (SLO)

| 지표 | 목표 | 측정 방법 |
|------|------|----------|
| 가용성 | 99.9% | Uptime Check |
| P50 응답 시간 | < 2초 | Cloud Trace |
| P99 응답 시간 | < 10초 | Cloud Trace |
| 에러율 | < 1% | Cloud Monitoring |

---

## 2. 일일 운영 작업

### 2.1 모닝 체크리스트

```
□ 대시보드 확인
  - 요청량 정상 여부
  - 에러율 확인
  - 지연 시간 확인

□ 알림 확인
  - 야간 알림 검토
  - 미해결 인시던트 확인

□ 로그 확인
  - 에러 로그 검토
  - 비정상 패턴 확인
```

### 2.2 대시보드 확인

**Cloud Console 접속:**
```
https://console.cloud.google.com/monitoring/dashboards?project=YOUR_PROJECT_ID
```

**주요 확인 항목:**

| 패널 | 정상 범위 | 주의 필요 |
|------|----------|----------|
| 요청 수 | 예상 트래픽 ±20% | 급격한 증가/감소 |
| 에러율 | < 1% | > 5% |
| P50 지연 | < 2초 | > 3초 |
| P99 지연 | < 10초 | > 15초 |
| 활성 세션 | 모니터링 | 급격한 증가 |
| 토큰 사용량 | 예산 내 | 예산 초과 |

### 2.3 로그 모니터링

```bash
# 최근 에러 확인
gcloud logging read \
  'resource.type="aiplatform.googleapis.com/AgentEngine" severity>=ERROR' \
  --limit=20 \
  --project=YOUR_PROJECT_ID

# 특정 시간대 로그
gcloud logging read \
  'timestamp>="2026-02-03T00:00:00Z" AND timestamp<="2026-02-03T12:00:00Z"' \
  --limit=100

# 특정 사용자 로그
gcloud logging read \
  'jsonPayload.user_id="user123"' \
  --limit=50
```

---

## 3. 주간 운영 작업

### 3.1 주간 체크리스트

```
□ 성능 리뷰
  - 주간 응답 시간 트렌드
  - 피크 타임 분석

□ 비용 분석
  - Gemini API 사용량
  - Agent Engine 비용
  - Session/Memory 비용

□ 버전 관리
  - 현재 버전 확인
  - 대기 중인 업데이트

□ 보안 검토
  - 접근 로그 확인
  - 권한 거부 로그 확인
```

### 3.2 성능 분석

```bash
# 주간 지연 시간 분석
gcloud logging read \
  'jsonPayload.latency_ms>0' \
  --format='value(jsonPayload.latency_ms)' \
  --project=YOUR_PROJECT_ID | \
  sort -n | \
  awk 'BEGIN{c=0} {a[c++]=$1} END{
    print "P50:", a[int(c*0.50)], "ms";
    print "P90:", a[int(c*0.90)], "ms";
    print "P99:", a[int(c*0.99)], "ms";
  }'
```

### 3.3 비용 모니터링

```bash
# Cloud Console에서 비용 확인
# Billing > Reports > Filter by Service

# 예상 비용 계산
# - Agent Engine Runtime: vCPU-초 × $0.000044
# - Gemini API: 토큰 수 × 요금
# - Sessions/Memory: 사용량 기반
```

---

## 4. 월간 운영 작업

### 4.1 월간 체크리스트

```
□ SLA/SLO 리뷰
  - 월간 가용성 계산
  - SLO 달성 여부

□ IAM 권한 검토
  - 불필요한 권한 제거
  - 서비스 계정 검토

□ 버전 정리
  - 오래된 버전 삭제
  - 레지스트리 정리

□ 문서 업데이트
  - 운영 문서 검토
  - 런북 업데이트

□ 용량 계획
  - 다음 달 예상 트래픽
  - 스케일링 설정 검토
```

### 4.2 가용성 계산

```bash
# 월간 다운타임 계산
# 99.9% SLA = 월 43.2분 이하 다운타임

# Uptime Check 결과 확인
gcloud monitoring uptime list-checks --project=YOUR_PROJECT_ID
```

### 4.3 버전 정리

```bash
# 현재 버전 확인
cat versions/registry.yaml

# 오래된 버전 목록
python scripts/version/rollback.py --env production --list

# 정리 (최근 5개 유지)
python scripts/version/cleanup.py --env production --keep 5
```

---

## 5. 인시던트 대응

### 5.1 인시던트 분류

| 레벨 | 조건 | 대응 시간 | 예시 |
|------|------|----------|------|
| P1 (Critical) | 서비스 완전 중단 | 15분 이내 | 전체 장애 |
| P2 (High) | 주요 기능 장애 | 1시간 이내 | 에러율 > 10% |
| P3 (Medium) | 부분 장애 | 4시간 이내 | 지연 증가 |
| P4 (Low) | 경미한 문제 | 24시간 이내 | 간헐적 오류 |

### 5.2 인시던트 대응 프로세스

```
1. 감지 (Detection)
   ↓
2. 분류 (Classification)
   ↓
3. 커뮤니케이션 (Communication)
   ↓
4. 조사 (Investigation)
   ↓
5. 완화 (Mitigation)
   ↓
6. 해결 (Resolution)
   ↓
7. 사후 분석 (Post-mortem)
```

### 5.3 빠른 대응 명령어

```bash
# 1. 현재 상태 확인
gcloud ai agent-engines describe AGENT_ENGINE_ID

# 2. 최근 에러 확인
gcloud logging read 'severity>=ERROR' --limit=20

# 3. 즉시 롤백
python scripts/version/rollback.py \
  --env production \
  --reason "Incident response"

# 4. 알림 상태 확인
gcloud alpha monitoring policies list
```

### 5.4 에스컬레이션 경로

| 시간 | 액션 |
|------|------|
| 0-15분 | 온콜 엔지니어가 조사 및 초기 대응 |
| 15-30분 | 팀 리드 알림, 추가 리소스 요청 |
| 30분-1시간 | 매니저 알림, 전체 팀 동원 |
| 1시간+ | 경영진 보고, GCP 지원 케이스 |

---

## 6. 롤백 절차

### 6.1 롤백 결정 기준

다음 조건 중 하나라도 해당되면 롤백을 고려:

- 에러율 > 10% (5분 이상 지속)
- P99 지연 > 30초
- 핵심 기능 장애
- 데이터 손상 위험

### 6.2 롤백 실행

```bash
# 1. 현재 버전 확인
python scripts/version/rollback.py --env production --list

# 2. 롤백 실행
python scripts/version/rollback.py \
  --env production \
  --reason "High error rate after deployment v1.1.0"

# 3. 롤백 확인
gcloud ai agent-engines describe AGENT_ENGINE_ID

# 4. 모니터링
# 대시보드에서 에러율/지연시간 정상화 확인
```

### 6.3 롤백 후 작업

1. 인시던트 기록 작성
2. 원인 분석
3. 수정 개발
4. 테스트 강화
5. 재배포

---

## 7. 스케일링

### 7.1 현재 설정 확인

```bash
# Agent Engine 상세 정보
gcloud ai agent-engines describe AGENT_ENGINE_ID --format=yaml
```

### 7.2 스케일링 설정

```yaml
# Agent Engine 스케일링 설정
scaling:
  min_instances: 1      # Production: 최소 1
  max_instances: 100    # 최대 100
  target_cpu: 0.7       # CPU 70%에서 스케일업
```

### 7.3 수동 스케일링

특별 이벤트 (프로모션 등) 대비:

```bash
# 사전 스케일업
# min_instances 증가로 웜업 시간 단축

# 이벤트 종료 후 복원
# min_instances 원복
```

---

## 8. 백업 및 복구

### 8.1 백업 대상

| 항목 | 백업 방법 | 주기 |
|------|----------|------|
| 버전 레지스트리 | Git 저장소 | 코드 푸시 시 |
| 설정 파일 | Git 저장소 | 코드 푸시 시 |
| 알림 정책 | YAML export | 변경 시 |

### 8.2 Session/Memory

- Agent Engine이 관리
- TTL 기반 자동 만료
- 별도 백업 불필요 (또는 불가)

### 8.3 설정 백업

```bash
# 알림 정책 백업
gcloud alpha monitoring policies list --format=yaml > backup/alerts.yaml

# 대시보드 백업
gcloud monitoring dashboards describe DASHBOARD_ID --format=yaml > backup/dashboard.yaml
```

---

## 9. 보안 운영

### 9.1 정기 보안 검토

```bash
# IAM 권한 확인
gcloud projects get-iam-policy YOUR_PROJECT_ID

# 서비스 계정 목록
gcloud iam service-accounts list

# 감사 로그 확인
gcloud logging read \
  'protoPayload.serviceName="aiplatform.googleapis.com"' \
  --limit=100
```

### 9.2 보안 인시던트 대응

1. **권한 거부 급증 감지**
   ```bash
   gcloud logging read \
     'protoPayload.status.code="7"' \
     --limit=50
   ```

2. **의심스러운 활동 조사**
   ```bash
   gcloud logging read \
     'protoPayload.authenticationInfo.principalEmail!=""' \
     --format='value(protoPayload.authenticationInfo.principalEmail)' | \
     sort | uniq -c | sort -rn
   ```

3. **필요시 계정 비활성화**
   ```bash
   gcloud iam service-accounts disable SA_EMAIL
   ```

### 9.3 키 로테이션

```bash
# 서비스 계정 키 목록
gcloud iam service-accounts keys list --iam-account=SA_EMAIL

# 새 키 생성
gcloud iam service-accounts keys create new-key.json --iam-account=SA_EMAIL

# 이전 키 삭제
gcloud iam service-accounts keys delete KEY_ID --iam-account=SA_EMAIL
```

---

## 10. 비용 관리

### 10.1 비용 구성

| 항목 | 과금 기준 | 예상 비용 |
|------|----------|----------|
| Agent Engine Runtime | vCPU-초 × $0.000044 | ~$150/월 |
| Gemini API | 토큰 수 | ~$300/월 |
| Sessions | 세션 수 (TBD) | TBD |
| Memory Bank | 메모리 수 (TBD) | TBD |
| Cloud Logging | 저장량 | ~$50/월 |
| Cloud Monitoring | 메트릭 | ~$50/월 |

### 10.2 비용 최적화

1. **모델 선택**
   - 간단한 쿼리: Gemini Flash (저렴)
   - 복잡한 쿼리: Gemini Pro

2. **토큰 최적화**
   - 프롬프트 길이 최소화
   - 불필요한 컨텍스트 제거

3. **세션/메모리 정리**
   - 적절한 TTL 설정
   - 불필요한 데이터 정리

### 10.3 예산 알림

```bash
# 예산 알림 설정 (Cloud Console)
# Billing > Budgets & alerts > Create budget
```

---

## 11. 문서 및 연락처

### 11.1 관련 문서

| 문서 | 설명 |
|------|------|
| [RUNBOOK.md](./RUNBOOK.md) | 알림 대응 절차 |
| [DEPLOYMENT.md](./DEPLOYMENT.md) | 배포 가이드 |
| [SECURITY.md](./SECURITY.md) | 보안 가이드 |
| [API_REFERENCE.md](./API_REFERENCE.md) | API 문서 |

### 11.2 연락처

| 역할 | 연락처 |
|------|--------|
| 온콜 엔지니어 | oncall@your-domain.com |
| 플랫폼 팀 | platform@your-domain.com |
| 보안 팀 | security@your-domain.com |
| GCP 지원 | support.google.com |

### 11.3 유용한 링크

- [Cloud Console](https://console.cloud.google.com)
- [Cloud Monitoring](https://console.cloud.google.com/monitoring)
- [Cloud Logging](https://console.cloud.google.com/logs)
- [GCP Status](https://status.cloud.google.com)
