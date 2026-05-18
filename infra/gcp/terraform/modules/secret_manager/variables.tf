variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "app_env" {
  description = "Application environment (local | staging | prod)"
  type        = string
}

variable "openai_api_key" {
  description = "OpenAI API key — stored as a Secret Manager secret version"
  type        = string
  sensitive   = true
  default     = ""
}

variable "create_openai_secret" {
  description = "Whether to create the OpenAI API key secret (set false to manage it outside Terraform)"
  type        = bool
  default     = true
}

variable "secret_replication_locations" {
  description = "List of GCP regions for secret replication (empty = automatic)"
  type        = list(string)
  default     = []
}
