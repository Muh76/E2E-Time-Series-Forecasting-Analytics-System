locals {
  etl_job_name      = "forecasting-etl-daily-${var.app_env}"
  training_job_name = "forecasting-train-weekly-${var.app_env}"
}

# ---------------------------------------------------------------------------
# Daily ETL job
# Triggers the ETL pipeline endpoint on the Cloud Run backend each morning.
# The backend exposes POST /api/v1/admin/etl/run (internal, SA-authenticated).
# ---------------------------------------------------------------------------

resource "google_cloud_scheduler_job" "etl_daily" {
  project   = var.project_id
  region    = var.region
  name      = local.etl_job_name
  schedule  = var.etl_schedule
  time_zone = var.timezone

  description = "Daily ETL pipeline: ingest raw data → validate → clean → write parquet"

  attempt_deadline = var.attempt_deadline

  retry_config {
    retry_count          = var.retry_count
    min_backoff_duration = "300s"
    max_backoff_duration = "3600s"
    max_doublings        = 3
  }

  http_target {
    http_method = "POST"
    uri         = "${var.backend_service_url}/api/v1/admin/etl/run"

    headers = {
      "Content-Type" = "application/json"
    }

    body = base64encode(jsonencode({
      mode    = "rossmann"
      dry_run = false
    }))

    oidc_token {
      service_account_email = var.scheduler_sa_email
      audience              = var.backend_service_url
    }
  }
}

# ---------------------------------------------------------------------------
# Weekly model training job
# Triggers full model retraining after the ETL output is fresh.
# ---------------------------------------------------------------------------

resource "google_cloud_scheduler_job" "training_weekly" {
  project   = var.project_id
  region    = var.region
  name      = local.training_job_name
  schedule  = var.training_schedule
  time_zone = var.timezone

  description = "Weekly model training: feature engineering → LightGBM fit → save artifacts"

  attempt_deadline = var.attempt_deadline

  retry_config {
    retry_count          = var.retry_count
    min_backoff_duration = "600s"
    max_backoff_duration = "7200s"
    max_doublings        = 2
  }

  http_target {
    http_method = "POST"
    uri         = "${var.backend_service_url}/api/v1/admin/training/run"

    headers = {
      "Content-Type" = "application/json"
    }

    body = base64encode(jsonencode({
      env        = var.app_env
      save_artifacts = true
    }))

    oidc_token {
      service_account_email = var.scheduler_sa_email
      audience              = var.backend_service_url
    }
  }
}
