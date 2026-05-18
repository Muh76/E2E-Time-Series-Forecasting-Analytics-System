variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for the Cloud Run service"
  type        = string
}

variable "app_env" {
  description = "Application environment (local | staging | prod)"
  type        = string
}

variable "service_name" {
  description = "Cloud Run service name (default: forecasting-backend-<env>)"
  type        = string
  default     = ""
}

variable "image" {
  description = "Full Docker image URI to deploy"
  type        = string
}

variable "service_account_email" {
  description = "Service account email for the Cloud Run service identity"
  type        = string
}

variable "cpu" {
  description = "CPU limit per container instance (e.g. '1', '2')"
  type        = string
  default     = "1"
}

variable "memory" {
  description = "Memory limit per container instance (e.g. '2Gi', '4Gi')"
  type        = string
  default     = "2Gi"
}

variable "min_instances" {
  description = "Minimum number of instances (0 = scale to zero)"
  type        = number
  default     = 0
}

variable "max_instances" {
  description = "Maximum number of instances"
  type        = number
  default     = 4
}

variable "concurrency" {
  description = "Maximum concurrent requests per instance"
  type        = number
  default     = 80
}

variable "timeout_seconds" {
  description = "Request timeout in seconds"
  type        = number
  default     = 300
}

variable "allow_unauthenticated" {
  description = "Allow public (unauthenticated) access to the service"
  type        = bool
  default     = true
}

variable "artifacts_bucket_name" {
  description = "GCS artifacts bucket name injected as env var GCS_ARTIFACTS_BUCKET"
  type        = string
}

variable "data_bucket_name" {
  description = "GCS data bucket name injected as env var GCS_DATA_BUCKET"
  type        = string
}

variable "rag_bucket_name" {
  description = "GCS RAG documents bucket name injected as env var GCS_RAG_BUCKET"
  type        = string
}

variable "openai_secret_version_name" {
  description = "Full Secret Manager version name for OPENAI_API_KEY (empty = not injected)"
  type        = string
  default     = ""
}

variable "extra_env_vars" {
  description = "Additional plain-text environment variables to inject into the container"
  type        = map(string)
  default     = {}
}
