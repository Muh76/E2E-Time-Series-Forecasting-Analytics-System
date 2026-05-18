locals {
  sa_account_id = "forecasting-backend-${var.app_env}"
}

# ---------------------------------------------------------------------------
# Service account for the Cloud Run backend
# ---------------------------------------------------------------------------

resource "google_service_account" "backend" {
  project      = var.project_id
  account_id   = local.sa_account_id
  display_name = "Forecasting Backend SA (${var.app_env})"
  description  = "Identity used by the Cloud Run backend to access GCS, Secret Manager, and Artifact Registry"
}

# ---------------------------------------------------------------------------
# GCS bucket access
# ---------------------------------------------------------------------------

resource "google_storage_bucket_iam_member" "backend_artifacts_reader" {
  bucket = var.artifacts_bucket_name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.backend.email}"
}

resource "google_storage_bucket_iam_member" "backend_data_reader" {
  bucket = var.data_bucket_name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.backend.email}"
}

resource "google_storage_bucket_iam_member" "backend_rag_reader" {
  bucket = var.rag_bucket_name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.backend.email}"
}

# ---------------------------------------------------------------------------
# Secret Manager access
# ---------------------------------------------------------------------------

resource "google_project_iam_member" "backend_secret_accessor" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.backend.email}"
}

# ---------------------------------------------------------------------------
# Artifact Registry — pull images at Cloud Run deploy time
# ---------------------------------------------------------------------------

resource "google_artifact_registry_repository_iam_member" "backend_ar_reader" {
  project    = var.project_id
  location   = var.region
  repository = var.artifact_registry_repository_id
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${google_service_account.backend.email}"
}

# ---------------------------------------------------------------------------
# Cloud Run invoker — allow the Scheduler SA to trigger Cloud Run jobs
# ---------------------------------------------------------------------------

resource "google_service_account" "scheduler" {
  project      = var.project_id
  account_id   = "forecasting-scheduler-${var.app_env}"
  display_name = "Forecasting Cloud Scheduler SA (${var.app_env})"
  description  = "Identity used by Cloud Scheduler to invoke Cloud Run Jobs (ETL, training)"
}

resource "google_project_iam_member" "scheduler_run_invoker" {
  project = var.project_id
  role    = "roles/run.invoker"
  member  = "serviceAccount:${google_service_account.scheduler.email}"
}
