# ---------------------------------------------------------------------------
# Cloud Run
# ---------------------------------------------------------------------------

output "backend_service_url" {
  description = "Public HTTPS URL of the Cloud Run backend service"
  value       = module.cloud_run.service_url
}

output "backend_service_name" {
  description = "Cloud Run service name"
  value       = module.cloud_run.service_name
}

# ---------------------------------------------------------------------------
# GCS
# ---------------------------------------------------------------------------

output "artifacts_bucket_name" {
  description = "GCS bucket name for model artifacts"
  value       = module.gcs.artifacts_bucket_name
}

output "artifacts_bucket_url" {
  description = "GCS URI for model artifacts bucket"
  value       = module.gcs.artifacts_bucket_url
}

output "data_bucket_name" {
  description = "GCS bucket name for raw and processed data"
  value       = module.gcs.data_bucket_name
}

output "rag_bucket_name" {
  description = "GCS bucket name for RAG knowledge base documents"
  value       = module.gcs.rag_bucket_name
}

# ---------------------------------------------------------------------------
# Artifact Registry
# ---------------------------------------------------------------------------

output "docker_image_prefix" {
  description = "Image path prefix for building and pushing Docker images (append /<name>:<tag>)"
  value       = module.artifact_registry.image_prefix
}

output "registry_hostname" {
  description = "Docker registry hostname for 'docker login'"
  value       = module.artifact_registry.registry_hostname
}

# ---------------------------------------------------------------------------
# IAM
# ---------------------------------------------------------------------------

output "backend_service_account_email" {
  description = "Service account email used by the Cloud Run backend"
  value       = module.iam.backend_service_account_email
}

output "scheduler_service_account_email" {
  description = "Service account email used by Cloud Scheduler"
  value       = module.iam.scheduler_service_account_email
}

# ---------------------------------------------------------------------------
# Cloud Scheduler
# ---------------------------------------------------------------------------

output "etl_job_name" {
  description = "Name of the daily ETL Cloud Scheduler job"
  value       = module.cloud_scheduler.etl_job_name
}

output "training_job_name" {
  description = "Name of the weekly training Cloud Scheduler job"
  value       = module.cloud_scheduler.training_job_name
}
