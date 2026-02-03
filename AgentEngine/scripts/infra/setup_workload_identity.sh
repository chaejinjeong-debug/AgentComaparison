#!/bin/bash
# Setup Workload Identity Federation for GitHub Actions
# This script creates the necessary GCP resources for GitHub Actions authentication

set -euo pipefail

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-}"
GITHUB_ORG="${GITHUB_ORG:-}"
GITHUB_REPO="${GITHUB_REPO:-}"
POOL_NAME="github-actions-pool"
PROVIDER_NAME="github-provider"
SERVICE_ACCOUNT_NAME="github-actions-deployer"
LOCATION="global"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."

    if [ -z "$PROJECT_ID" ]; then
        log_error "GCP_PROJECT_ID environment variable is not set"
        echo "Usage: GCP_PROJECT_ID=<project-id> GITHUB_ORG=<org> GITHUB_REPO=<repo> $0"
        exit 1
    fi

    if [ -z "$GITHUB_ORG" ]; then
        log_error "GITHUB_ORG environment variable is not set"
        exit 1
    fi

    if [ -z "$GITHUB_REPO" ]; then
        log_error "GITHUB_REPO environment variable is not set"
        exit 1
    fi

    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI is not installed"
        exit 1
    fi

    log_info "Prerequisites check passed"
}

enable_apis() {
    log_info "Enabling required APIs..."

    gcloud services enable \
        iam.googleapis.com \
        iamcredentials.googleapis.com \
        cloudresourcemanager.googleapis.com \
        aiplatform.googleapis.com \
        storage.googleapis.com \
        --project="$PROJECT_ID"

    log_info "APIs enabled successfully"
}

create_service_account() {
    log_info "Creating service account..."

    SA_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

    # Check if service account exists
    if gcloud iam service-accounts describe "$SA_EMAIL" --project="$PROJECT_ID" &> /dev/null; then
        log_warn "Service account $SA_EMAIL already exists"
    else
        gcloud iam service-accounts create "$SERVICE_ACCOUNT_NAME" \
            --display-name="GitHub Actions Deployer" \
            --description="Service account for GitHub Actions CI/CD deployments" \
            --project="$PROJECT_ID"
        log_info "Service account created: $SA_EMAIL"
    fi

    # Grant necessary roles
    log_info "Granting IAM roles to service account..."

    ROLES=(
        "roles/aiplatform.admin"
        "roles/storage.admin"
        "roles/iam.serviceAccountUser"
    )

    for ROLE in "${ROLES[@]}"; do
        gcloud projects add-iam-policy-binding "$PROJECT_ID" \
            --member="serviceAccount:$SA_EMAIL" \
            --role="$ROLE" \
            --condition=None \
            --quiet
        log_info "Granted role: $ROLE"
    done
}

create_workload_identity_pool() {
    log_info "Creating Workload Identity Pool..."

    POOL_ID="projects/${PROJECT_ID}/locations/${LOCATION}/workloadIdentityPools/${POOL_NAME}"

    # Check if pool exists
    if gcloud iam workload-identity-pools describe "$POOL_NAME" \
        --location="$LOCATION" \
        --project="$PROJECT_ID" &> /dev/null; then
        log_warn "Workload Identity Pool $POOL_NAME already exists"
    else
        gcloud iam workload-identity-pools create "$POOL_NAME" \
            --location="$LOCATION" \
            --display-name="GitHub Actions Pool" \
            --description="Workload Identity Pool for GitHub Actions" \
            --project="$PROJECT_ID"
        log_info "Workload Identity Pool created: $POOL_NAME"
    fi
}

create_oidc_provider() {
    log_info "Creating OIDC Provider..."

    # Check if provider exists
    if gcloud iam workload-identity-pools providers describe "$PROVIDER_NAME" \
        --workload-identity-pool="$POOL_NAME" \
        --location="$LOCATION" \
        --project="$PROJECT_ID" &> /dev/null; then
        log_warn "OIDC Provider $PROVIDER_NAME already exists"
    else
        gcloud iam workload-identity-pools providers create-oidc "$PROVIDER_NAME" \
            --workload-identity-pool="$POOL_NAME" \
            --location="$LOCATION" \
            --issuer-uri="https://token.actions.githubusercontent.com" \
            --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository,attribute.repository_owner=assertion.repository_owner" \
            --attribute-condition="assertion.repository_owner == '${GITHUB_ORG}'" \
            --project="$PROJECT_ID"
        log_info "OIDC Provider created: $PROVIDER_NAME"
    fi
}

bind_service_account() {
    log_info "Binding service account to Workload Identity Pool..."

    SA_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
    POOL_ID="projects/${PROJECT_ID}/locations/${LOCATION}/workloadIdentityPools/${POOL_NAME}"

    # Allow the GitHub Actions from specific repository to impersonate the service account
    gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
        --project="$PROJECT_ID" \
        --role="roles/iam.workloadIdentityUser" \
        --member="principalSet://iam.googleapis.com/${POOL_ID}/attribute.repository/${GITHUB_ORG}/${GITHUB_REPO}"

    log_info "Service account binding completed"
}

print_github_secrets() {
    log_info "GitHub Actions configuration completed!"
    echo ""
    echo "=========================================="
    echo "  GitHub Secrets to Configure"
    echo "=========================================="
    echo ""

    PROVIDER_FULL_PATH="projects/${PROJECT_ID}/locations/${LOCATION}/workloadIdentityPools/${POOL_NAME}/providers/${PROVIDER_NAME}"
    SA_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

    echo "Add the following secrets to your GitHub repository:"
    echo ""
    echo "GCP_PROJECT_ID:"
    echo "  ${PROJECT_ID}"
    echo ""
    echo "GCP_WORKLOAD_IDENTITY_PROVIDER:"
    echo "  ${PROVIDER_FULL_PATH}"
    echo ""
    echo "GCP_SERVICE_ACCOUNT:"
    echo "  ${SA_EMAIL}"
    echo ""
    echo "=========================================="
    echo ""
    echo "To add secrets via GitHub CLI:"
    echo ""
    echo "gh secret set GCP_PROJECT_ID --body \"${PROJECT_ID}\""
    echo "gh secret set GCP_WORKLOAD_IDENTITY_PROVIDER --body \"${PROVIDER_FULL_PATH}\""
    echo "gh secret set GCP_SERVICE_ACCOUNT --body \"${SA_EMAIL}\""
    echo ""
}

cleanup() {
    log_warn "Cleanup function called. To remove created resources, run:"
    echo ""
    echo "# Delete OIDC Provider"
    echo "gcloud iam workload-identity-pools providers delete $PROVIDER_NAME \\"
    echo "  --workload-identity-pool=$POOL_NAME \\"
    echo "  --location=$LOCATION \\"
    echo "  --project=$PROJECT_ID"
    echo ""
    echo "# Delete Workload Identity Pool"
    echo "gcloud iam workload-identity-pools delete $POOL_NAME \\"
    echo "  --location=$LOCATION \\"
    echo "  --project=$PROJECT_ID"
    echo ""
    echo "# Delete Service Account"
    echo "gcloud iam service-accounts delete ${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com \\"
    echo "  --project=$PROJECT_ID"
}

main() {
    echo ""
    echo "=========================================="
    echo "  Workload Identity Federation Setup"
    echo "  for GitHub Actions"
    echo "=========================================="
    echo ""

    check_prerequisites
    enable_apis
    create_service_account
    create_workload_identity_pool
    create_oidc_provider
    bind_service_account
    print_github_secrets

    log_info "Setup completed successfully!"
}

# Handle cleanup flag
if [[ "${1:-}" == "--cleanup" ]]; then
    cleanup
    exit 0
fi

main "$@"
