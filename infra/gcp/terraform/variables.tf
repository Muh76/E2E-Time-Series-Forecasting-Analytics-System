variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for Cloud Run and GCS"
  type        = string
  default     = "europe-west1"
}

variable "app_env" {
  description = "Application environment (local | staging | prod)"
  type        = string
  default     = "prod"
  validation {
    condition     = contains(["local", "staging", "prod"], var.app_env)
    error_message = "app_env must be one of: local, staging, prod."
  }
}

variable "backend_image" {
  description = "Full Docker image URI for the backend service (e.g. europe-west1-docker.pkg.dev/PROJECT/repo/backend:tag)"
  type        = string
}

variable "backend_cpu" {
  description = "CPU allocation for the Cloud Run backend service"
  type        = string
  default     = "1"
}

variable "backend_memory" {
  description = "Memory allocation for the Cloud Run backend service"
  type        = string
  default     = "2Gi"
}

variable "backend_min_instances" {
  description = "Minimum number of backend Cloud Run instances (0 = scale to zero)"
  type        = number
  default     = 0
}

variable "backend_max_instances" {
  description = "Maximum number of backend Cloud Run instances"
  type        = number
  default     = 4
}

variable "backend_concurrency" {
  description = "Maximum concurrent requests per backend instance"
  type        = number
  default     = 80
}

variable "openai_secret_version" {
  description = "Secret Manager secret version for OPENAI_API_KEY (e.g. projects/PROJECT/secrets/openai-api-key/versions/latest)"
  type        = string
  default     = ""
}

variable "artifacts_bucket_location" {
  description = "GCS multi-region location for the artifacts bucket"
  type        = string
  default     = "EU"
}

variable "data_bucket_location" {
  description = "GCS multi-region location for the data bucket"
  type        = string
  default     = "EU"
}
