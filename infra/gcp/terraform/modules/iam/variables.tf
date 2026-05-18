variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "app_env" {
  description = "Application environment (local | staging | prod)"
  type        = string
}

variable "artifacts_bucket_name" {
  description = "Name of the GCS artifacts bucket (grants objectViewer to backend SA)"
  type        = string
}

variable "data_bucket_name" {
  description = "Name of the GCS data bucket (grants objectViewer to backend SA)"
  type        = string
}

variable "rag_bucket_name" {
  description = "Name of the GCS RAG documents bucket (grants objectViewer to backend SA)"
  type        = string
}

variable "artifact_registry_repository_id" {
  description = "Artifact Registry repository ID (grants reader to Cloud Run SA)"
  type        = string
}

variable "region" {
  description = "GCP region (needed for Artifact Registry IAM binding)"
  type        = string
}
