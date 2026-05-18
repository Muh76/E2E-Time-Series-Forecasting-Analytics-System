variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "app_env" {
  description = "Application environment (local | staging | prod)"
  type        = string
}

variable "location" {
  description = "GCS multi-region location (EU | US | ASIA)"
  type        = string
  default     = "EU"
}

variable "artifacts_bucket_name" {
  description = "Override the generated artifacts bucket name (leave empty to auto-generate)"
  type        = string
  default     = ""
}

variable "data_bucket_name" {
  description = "Override the generated data bucket name (leave empty to auto-generate)"
  type        = string
  default     = ""
}

variable "rag_bucket_name" {
  description = "Override the generated RAG documents bucket name (leave empty to auto-generate)"
  type        = string
  default     = ""
}

variable "artifacts_versioning_enabled" {
  description = "Enable object versioning on the artifacts bucket"
  type        = bool
  default     = true
}

variable "artifacts_keep_versions" {
  description = "Number of non-current object versions to keep in the artifacts bucket"
  type        = number
  default     = 5
}

variable "force_destroy" {
  description = "Allow Terraform to destroy non-empty buckets (use false in prod)"
  type        = bool
  default     = false
}
