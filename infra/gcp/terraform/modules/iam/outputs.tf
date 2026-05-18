output "backend_service_account_email" {
  description = "Email of the Cloud Run backend service account"
  value       = google_service_account.backend.email
}

output "backend_service_account_id" {
  description = "Full resource ID of the backend service account"
  value       = google_service_account.backend.id
}

output "scheduler_service_account_email" {
  description = "Email of the Cloud Scheduler service account"
  value       = google_service_account.scheduler.email
}
