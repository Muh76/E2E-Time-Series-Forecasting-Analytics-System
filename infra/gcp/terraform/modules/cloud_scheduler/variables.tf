variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for Cloud Scheduler jobs"
  type        = string
}

variable "app_env" {
  description = "Application environment (local | staging | prod)"
  type        = string
}

variable "scheduler_sa_email" {
  description = "Service account email that Cloud Scheduler uses to invoke Cloud Run Jobs"
  type        = string
}

variable "backend_service_url" {
  description = "Base URL of the Cloud Run backend service (e.g. https://forecasting-backend-xxx-ew.a.run.app)"
  type        = string
}

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

variable "timezone" {
  description = "Timezone for all Cloud Scheduler jobs"
  type        = string
  default     = "UTC"
}

variable "attempt_deadline" {
  description = "Maximum time to wait for a job attempt to complete"
  type        = string
  default     = "1800s"
}

variable "retry_count" {
  description = "Maximum number of retries on job failure"
  type        = number
  default     = 1
}
