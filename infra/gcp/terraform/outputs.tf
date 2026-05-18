output "backend_service_url" {
  description = "Public URL of the Cloud Run backend service"
  value       = google_cloud_run_v2_service.backend.uri
}

output "artifacts_bucket_name" {
  description = "GCS bucket for model artifacts"
  value       = google_storage_bucket.artifacts.name
}

output "data_bucket_name" {
  description = "GCS bucket for raw and processed data"
  value       = google_storage_bucket.data.name
}

output "artifact_registry_repository" {
  description = "Artifact Registry repository URI for pushing backend images"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.backend.repository_id}"
}

output "backend_service_account_email" {
  description = "Service account email used by the Cloud Run backend"
  value       = google_service_account.backend.email
}
