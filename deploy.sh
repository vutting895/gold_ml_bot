#!/usr/bin/env bash
set -e

# ==================== CONFIGURATION ====================
REGION="asia-southeast1"                           # Singapore
REPOSITORY="gold-bot-repo"                         # ชื่อ Artifact Registry
JOB_NAME="gold-ml-scanner"                         # ชื่อ Cloud Run Job
SCHEDULER_NAME="gold-bot-5min-trigger"             # ชื่อ Cloud Scheduler Job
CRON_SCHEDULE="*/5 * * * 1-5"                      # รันทุก 5 นาที (จันทร์ - ศุกร์)
TIMEZONE="Asia/Bangkok"                            # เขตเวลาไทย

PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
    echo "❌ ไม่พบ GCP Project ID กรุณาเลือก Project ด้วยคำสั่ง: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
SERVICE_ACCOUNT="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${JOB_NAME}:latest"

echo "=================================================="
echo "🚀 STARTING GCP DEPLOYMENT PROCESS"
echo "📌 Project ID:     $PROJECT_ID"
echo "📌 Region:         $REGION"
echo "📌 Image URI:      $IMAGE_URI"
echo "=================================================="

# STEP 1: ENABLE GCP SERVICES
echo "🔌 [1/5] กำลังเปิดใช้งาน GCP APIs ที่จำเป็น..."
gcloud services enable     cloudbuild.googleapis.com     run.googleapis.com     cloudscheduler.googleapis.com     artifactregistry.googleapis.com     --project="$PROJECT_ID"

# STEP 2: ARTIFACT REGISTRY
echo "📦 [2/5] ตรวจสอบ Artifact Registry Repository..."
if ! gcloud artifacts repositories describe "$REPOSITORY" --location="$REGION" >/dev/null 2>&1; then
    echo "🔨 กำลังสร้าง Repository: $REPOSITORY..."
    gcloud artifacts repositories create "$REPOSITORY"         --repository-format=docker         --location="$REGION"         --description="Docker repository for Gold ML Bot"
else
    echo "✅ พบ Repository $REPOSITORY เรียบร้อยแล้ว"
fi

# STEP 3: BUILD & UPLOAD CONTAINER
echo "🏗️ [3/5] กำลัง Build และ Push Docker Image ขึ้น Cloud Build..."
gcloud builds submit --tag "$IMAGE_URI" .

# STEP 4: DEPLOY CLOUD RUN JOB
echo "🚀 [4/5] กำลังสร้าง / อัปเดต Cloud Run Job..."
gcloud run jobs deploy "$JOB_NAME"     --image="$IMAGE_URI"     --region="$REGION"     --tasks=1     --max-retries=1     --task-timeout=3m     --project="$PROJECT_ID"

# STEP 5: SETUP CLOUD SCHEDULER
echo "⏰ [5/5] กำลังตั้งค่า Cloud Scheduler และสิทธิ์ IAM..."

gcloud projects add-iam-policy-binding "$PROJECT_ID"     --member="serviceAccount:${SERVICE_ACCOUNT}"     --role="roles/run.invoker"     --quiet >/dev/null

JOB_URI="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${JOB_NAME}:run"

if gcloud scheduler jobs describe "$SCHEDULER_NAME" --location="$REGION" >/dev/null 2>&1; then
    echo "🔄 อัปเดต Cloud Scheduler ที่มีอยู่เดิม..."
    gcloud scheduler jobs update http "$SCHEDULER_NAME"         --location="$REGION"         --schedule="$CRON_SCHEDULE"         --time-zone="$TIMEZONE"         --uri="$JOB_URI"         --http-method=POST         --oauth-service-account-email="$SERVICE_ACCOUNT"
else
    echo "➕ สร้าง Cloud Scheduler ตัวใหม่..."
    gcloud scheduler jobs create http "$SCHEDULER_NAME"         --location="$REGION"         --schedule="$CRON_SCHEDULE"         --time-zone="$TIMEZONE"         --uri="$JOB_URI"         --http-method=POST         --oauth-service-account-email="$SERVICE_ACCOUNT"
fi

echo "=================================================="
echo "🎉 DEPLOYMENT COMPLETED SUCCESSFULLY!"
echo "=================================================="
