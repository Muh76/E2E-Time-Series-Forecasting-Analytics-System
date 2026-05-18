output "openai_secret_id" {
  description = "Full resource ID of the OpenAI API key secret"
  value       = var.create_openai_secret ? google_secret_manager_secret.openai_api_key[0].id : ""
}

output "openai_secret_name" {
  description = "Short name of the OpenAI API key secret (for Secret Manager references)"
  value       = var.create_openai_secret ? google_secret_manager_secret.openai_api_key[0].name : ""
}

output "openai_secret_version_name" {
  description = "Full resource name of the latest OpenAI secret version (for Cloud Run env injection)"
  value = (
    var.create_openai_secret && var.openai_api_key != ""
    ? "${google_secret_manager_secret.openai_api_key[0].name}/versions/latest"
    : ""
  )
}

output "app_config_secret_id" {
  description = "Full resource ID of the app configuration secret"
  value       = google_secret_manager_secret.app_config.id
}
