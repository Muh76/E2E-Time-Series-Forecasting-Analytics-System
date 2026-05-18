output "artifacts_bucket_name" {
  description = "Name of the GCS artifacts bucket"
  value       = google_storage_bucket.artifacts.name
}

output "artifacts_bucket_url" {
  description = "gs:// URI of the artifacts bucket"
  value       = "gs://${google_storage_bucket.artifacts.name}"
}

output "data_bucket_name" {
  description = "Name of the GCS data bucket"
  value       = google_storage_bucket.data.name
}

output "data_bucket_url" {
  description = "gs:// URI of the data bucket"
  value       = "gs://${google_storage_bucket.data.name}"
}

output "rag_bucket_name" {
  description = "Name of the GCS RAG documents bucket"
  value       = google_storage_bucket.rag.name
}

output "rag_bucket_url" {
  description = "gs:// URI of the RAG documents bucket"
  value       = "gs://${google_storage_bucket.rag.name}"
}
