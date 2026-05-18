output "service_name" {
  description = "Cloud Run service name"
  value       = google_cloud_run_v2_service.backend.name
}

output "service_url" {
  description = "Public HTTPS URL of the Cloud Run service"
  value       = google_cloud_run_v2_service.backend.uri
}

output "latest_revision" {
  description = "Name of the latest deployed revision"
  value       = google_cloud_run_v2_service.backend.latest_ready_revision
}
