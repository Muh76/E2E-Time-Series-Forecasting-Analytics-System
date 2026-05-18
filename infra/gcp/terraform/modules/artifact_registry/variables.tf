variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for the Artifact Registry repository"
  type        = string
}

variable "app_env" {
  description = "Application environment (local | staging | prod)"
  type        = string
}

variable "repository_id" {
  description = "Artifact Registry repository ID (default: forecasting-backend)"
  type        = string
  default     = "forecasting-backend"
}

variable "description" {
  description = "Human-readable description for the repository"
  type        = string
  default     = "Docker images for the E2E Forecasting backend service"
}

variable "cleanup_keep_tag_count" {
  description = "Number of tagged images to retain per image name (older ones are deleted)"
  type        = number
  default     = 10
}
