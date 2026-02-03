# Deployment Guide

AgentEngine 배포 가이드

## Overview

이 문서는 AgentEngine을 Staging 및 Production 환경에 배포하는 방법을 설명합니다.

---

## 환경 구성

### 환경 분리

| 환경 | 용도 | 프로젝트 |
|------|------|---------|
| Staging | 테스트, QA | `project-staging` |
| Production | 실제 서비스 | `project-production` |

### 환경 변수 설정

```bash
# Staging 환경
cp config/staging.env.example config/staging.env
# production.env 수정

# Production 환경
cp config/production.env.example config/production.env
# staging.env 수정
```

---

## 사전 요구사항

### 1. GCP 프로젝트 설정

```bash
# 프로젝트 선택
gcloud config set project YOUR_PROJECT_ID

# API 활성화
gcloud services enable \
  aiplatform.googleapis.com \
  cloudbuild.googleapis.com \
  cloudtrace.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com
```

### 2. 인증 설정

```bash
# 로컬 개발용
gcloud auth application-default login

# 서비스 계정 사용 (CI/CD)
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa-key.json
```

### 3. IAM 및 보안 설정

```bash
# IAM 설정
python scripts/infra/setup_iam.py --project YOUR_PROJECT_ID

# 알림 설정
python scripts/infra/setup_alerts.py --project YOUR_PROJECT_ID
```

---

## Staging 배포

### 1. 수동 배포

```bash
# 의존성 설치
cd AgentEngine
uv pip install -e ".[dev]"

# Staging 환경 로드
source config/staging.env

# 배포
python scripts/deploy_source.py --env staging
```

### 2. CI/CD 배포

main 브랜치에 푸시하면 자동으로 Staging에 배포됩니다:

```yaml
# cloudbuild.yaml의 deploy-staging 스텝
- id: 'deploy-staging'
  ...
  args:
    - '-c'
    - |
      if [ "$BRANCH_NAME" = "main" ]; then
        python scripts/deploy_source.py --env staging
      fi
```

### 3. 배포 확인

```bash
# Agent Engine 상태 확인
gcloud ai agent-engines list --project=YOUR_PROJECT_ID

# 테스트 쿼리
python scripts/chat_with_agent.py --env staging
```

---

## Production 배포

### 1. 버전 태그 생성

```bash
# 버전 태그 생성
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

### 2. 자동 배포

태그 푸시 시 자동으로 Production에 배포됩니다:

```yaml
# cloudbuild.yaml의 deploy-production 스텝
- id: 'deploy-production'
  ...
  args:
    - '-c'
    - |
      if [ -n "$TAG_NAME" ]; then
        python scripts/deploy_source.py --env production
        python scripts/version/register.py --version "$TAG_NAME"
      fi
```

### 3. Staging에서 승격

```bash
# Staging 버전을 Production으로 승격
python scripts/version/promote.py

# 확인
Staging version to promote: v1.0.0
Current production version: v0.9.0
Proceed with promotion? [y/N]: y
```

---

## 버전 관리

### 버전 등록

```bash
# 버전 등록
python scripts/version/register.py \
  --version v1.0.0 \
  --env production \
  --agent-id projects/.../agentEngines/ae-xxx
```

### 버전 조회

```bash
# 현재 버전 확인
cat versions/registry.yaml

# 버전 기록 확인
python scripts/version/rollback.py --env production --list
```

---

## 롤백

### 즉시 롤백

```bash
# 이전 버전으로 롤백
python scripts/version/rollback.py \
  --env production \
  --reason "Bug in v1.1.0"

# 특정 버전으로 롤백
python scripts/version/rollback.py \
  --env production \
  --target v1.0.0 \
  --reason "Rollback to stable version"
```

### 롤백 확인

```bash
# 롤백 기록 확인
python scripts/version/rollback.py --env production --history
```

---

## 품질 게이트

### CI/CD 파이프라인

배포 전 다음 검사를 통과해야 합니다:

1. **Lint** (ruff): 코드 스타일
2. **Type Check** (mypy): 타입 안전성
3. **Security** (bandit): 보안 취약점
4. **Test** (pytest): 단위/통합 테스트
5. **Coverage**: 80% 이상

### 평가 테스트

```bash
# 평가 실행
python scripts/evaluation/run_evaluation.py --threshold 0.85

# 성능 테스트
python scripts/evaluation/performance_test.py --duration 60
```

### 품질 목표

| 메트릭 | 목표 |
|--------|------|
| 테스트 커버리지 | >= 80% |
| 평가 정확도 | >= 85% |
| P50 지연시간 | < 2초 |
| P99 지연시간 | < 10초 |
| 에러율 | < 5% |

---

## 모니터링

### 대시보드

```bash
# 대시보드 URL
echo "https://console.cloud.google.com/monitoring/dashboards/..."
```

### 알림

다음 조건에서 알림이 발생합니다:

- 에러율 > 5%
- P99 지연시간 > 10초
- 5분간 요청 없음 (서비스 다운)

### 로그 확인

```bash
# 최근 에러 로그
gcloud logging read \
  'resource.type="aiplatform.googleapis.com/AgentEngine" severity>=ERROR' \
  --limit=10 \
  --project=YOUR_PROJECT_ID
```

---

## 트러블슈팅

### 배포 실패

1. Cloud Build 로그 확인
2. IAM 권한 확인
3. 네트워크 설정 확인

```bash
# Cloud Build 로그
gcloud builds list --limit=5
gcloud builds log BUILD_ID
```

### Agent Engine 오류

1. Agent Engine 상태 확인
2. 로그 확인
3. 리소스 제한 확인

```bash
# Agent Engine 상세 정보
gcloud ai agent-engines describe AGENT_ENGINE_ID
```

---

## 관련 문서

- [SECURITY.md](./SECURITY.md) - 보안 가이드
- [RUNBOOK.md](./RUNBOOK.md) - 운영 런북
- [cloudbuild.yaml](../cloudbuild.yaml) - CI/CD 파이프라인
- [config/](../config/) - 환경별 설정
