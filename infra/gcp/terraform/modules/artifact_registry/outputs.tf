output "repository_id" {
  description = "Artifact Registry repository ID"
  value       = google_artifact_registry_repository.backend.repository_id
}

output "repository_name" {
  description = "Full resource name of the Artifact Registry repository"
  value       = google_artifact_registry_repository.backend.name
}

output "registry_hostname" {
  description = "Docker registry hostname (for docker login and image tagging)"
  value       = "${var.region}-docker.pkg.dev"
}

output "image_prefix" {
  description = "Full image path prefix — append /<image-name>:<tag> to build image URIs"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.backend.repository_id}"
}
