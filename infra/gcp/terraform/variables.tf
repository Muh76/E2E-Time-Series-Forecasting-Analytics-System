variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for all regional resources (Cloud Run, Artifact Registry)"
  type        = string
  default     = "europe-west1"
}

variable "app_env" {
  description = "Application environment: local | staging | prod"
  type        = string
  default     = "prod"
  validation {
    condition     = contains(["local", "staging", "prod"], var.app_env)
    error_message = "app_env must be one of: local, staging, prod."
  }
}

# ---------------------------------------------------------------------------
# GCS
# ---------------------------------------------------------------------------

variable "gcs_location" {
  description = "GCS multi-region location for all buckets (EU | US | ASIA)"
  type        = string
  default     = "EU"
}

# ---------------------------------------------------------------------------
# Cloud Run
# ---------------------------------------------------------------------------

variable "backend_image" {
  description = "Full Docker image URI (e.g. europe-west1-docker.pkg.dev/PROJECT/repo/backend:tag)"
  type        = string
  default     = "europe-docker.pkg.dev/PROJECT_ID/forecasting-backend/backend:latest"
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
  description = "Minimum Cloud Run instances (0 = scale to zero)"
  type        = number
  default     = 0
}

variable "backend_max_instances" {
  description = "Maximum Cloud Run instances"
  type        = number
  default     = 4
}

variable "backend_concurrency" {
  description = "Max concurrent requests per Cloud Run instance"
  type        = number
  default     = 80
}

# ---------------------------------------------------------------------------
# Secret Manager
# ---------------------------------------------------------------------------

variable "openai_api_key" {
  description = "OpenAI API key value — stored in Secret Manager, never in state as plaintext"
  type        = string
  sensitive   = true
  default     = ""
}

# ---------------------------------------------------------------------------
# Cloud Scheduler
# ---------------------------------------------------------------------------

variable "etl_schedule" {
  description = "Cron schedule for the daily ETL job (UTC)"
  type        = string
  default     = "0 2 * * *"
}

variable "training_schedule" {
  description = "Cron schedule for the weekly model training job (UTC)"
  type        = string
  default     = "0 4 * * 0"
}

variable "scheduler_timezone" {
  description = "Timezone for Cloud Scheduler jobs"
  type        = string
  default     = "UTC"
}
