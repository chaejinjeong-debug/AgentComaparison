# Security Guide

AgentEngine 보안 가이드

## Overview

이 문서는 AgentEngine의 보안 아키텍처와 모범 사례를 설명합니다.

---

## 보안 원칙

### 1. 최소 권한 원칙 (Principle of Least Privilege)

모든 서비스 계정과 사용자에게 필요한 최소한의 권한만 부여합니다.

```yaml
# 예시: Runtime 서비스 계정
- name: agent-engine-runtime
  roles:
    - roles/aiplatform.user      # Agent 쿼리만 가능
    - roles/logging.logWriter    # 로그 쓰기만 가능
```

### 2. 심층 방어 (Defense in Depth)

여러 계층의 보안 제어를 적용합니다:

- **네트워크 계층**: VPC Service Controls
- **인증 계층**: IAM, Workload Identity
- **데이터 계층**: 암호화 (at rest, in transit)
- **애플리케이션 계층**: 입력 검증, 출력 필터링

### 3. 제로 트러스트 (Zero Trust)

모든 요청을 검증하고, 암묵적으로 신뢰하지 않습니다.

---

## IAM 구성

### 서비스 계정

| 서비스 계정 | 용도 | 권한 |
|------------|------|------|
| `agent-engine-runtime` | 프로덕션 실행 | aiplatform.user, logging.logWriter |
| `agent-engine-deployer` | CI/CD 배포 | aiplatform.admin |
| `agent-engine-admin` | 관리 작업 | 전체 관리자 |
| `agent-engine-viewer` | 모니터링 | 읽기 전용 |

### IAM 설정 적용

```bash
# IAM 설정 적용
python scripts/infra/setup_iam.py --project YOUR_PROJECT_ID

# 검증
gcloud projects get-iam-policy YOUR_PROJECT_ID
```

### 커스텀 역할

최소 권한을 위해 커스텀 역할을 사용합니다:

```yaml
- id: agentEngineInvoker
  permissions:
    - aiplatform.agentEngines.query
    - aiplatform.sessions.create
    - aiplatform.sessions.get
```

---

## VPC Service Controls

### 개요

VPC-SC는 데이터 유출을 방지하기 위한 보안 경계를 만듭니다.

### 보호되는 서비스

- `aiplatform.googleapis.com`
- `storage.googleapis.com`
- `logging.googleapis.com`
- `monitoring.googleapis.com`

### 설정

```bash
# VPC-SC 설정 적용
python scripts/infra/setup_vpc_sc.py \
  --organization YOUR_ORG_ID \
  --policy agent-engine-policy

# 검증
gcloud access-context-manager perimeters list
```

### 접근 수준

1. **corp_network**: 회사 네트워크에서만 접근
2. **trusted_identities**: 승인된 서비스 계정
3. **trusted_devices**: 관리되는 기기에서만

---

## 데이터 보안

### 전송 중 암호화 (In Transit)

- 모든 통신은 TLS 1.3 사용
- 인증서 자동 관리 (Cloud Load Balancer)

### 저장 시 암호화 (At Rest)

- 기본: Google 관리 암호화 키
- 선택: 고객 관리 암호화 키 (CMEK)

### CMEK 설정 (선택)

```bash
# 키 링 생성
gcloud kms keyrings create agent-engine-keyring \
  --location=asia-northeast3

# 키 생성
gcloud kms keys create agent-engine-key \
  --keyring=agent-engine-keyring \
  --location=asia-northeast3 \
  --purpose=encryption
```

---

## 인증 및 권한

### Workload Identity Federation

외부 시스템 (GitHub Actions)과의 안전한 통합:

```yaml
workload_identity:
  pool_name: agent-engine-wi-pool
  providers:
    - name: github-actions
      issuer_uri: "https://token.actions.githubusercontent.com"
```

### API 키 관리

- API 키 사용 금지 (서비스 계정 사용)
- 필요시 API 키 제한 설정

---

## 감사 로그

### Cloud Audit Logs 활성화

모든 관리 활동과 데이터 접근을 로깅합니다.

```bash
# 감사 로그 정책 확인
gcloud projects get-iam-policy YOUR_PROJECT_ID \
  --format=json | jq '.auditConfigs'
```

### 로그 필터

```
# 권한 거부 이벤트
protoPayload.status.code="7"

# Agent Engine 접근
resource.type="aiplatform.googleapis.com/AgentEngine"
```

---

## PII 처리

### 마스킹

민감한 정보는 로그에서 마스킹합니다:

```python
# 로그에서 PII 마스킹
logger.info(
    "user_query",
    user_id=mask_pii(user_id),
    query=mask_pii(query),
)
```

### GDPR 대응

Memory Bank 삭제 기능 (MB-006):

```python
# 사용자 데이터 삭제
memory_manager.delete_user_memories(user_id)
```

---

## 보안 체크리스트

### 배포 전

- [ ] 서비스 계정 최소 권한 확인
- [ ] VPC-SC 경계 설정
- [ ] 감사 로그 활성화
- [ ] 보안 스캔 (bandit) 통과

### 운영 중

- [ ] 정기적 IAM 검토
- [ ] 로그 모니터링
- [ ] 알림 설정 확인
- [ ] 취약점 패치 적용

### 인시던트 대응

- [ ] 보안 인시던트 절차 숙지
- [ ] 연락처 목록 최신화
- [ ] 롤백 절차 테스트

---

## 관련 문서

- [DEPLOYMENT.md](./DEPLOYMENT.md) - 배포 가이드
- [RUNBOOK.md](./RUNBOOK.md) - 운영 런북
- [infra/iam/](../infra/iam/) - IAM 설정
- [infra/vpc-sc/](../infra/vpc-sc/) - VPC-SC 설정
