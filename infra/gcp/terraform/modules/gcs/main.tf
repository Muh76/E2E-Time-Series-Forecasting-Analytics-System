locals {
  artifacts_bucket = var.artifacts_bucket_name != "" ? var.artifacts_bucket_name : "${var.project_id}-forecasting-artifacts-${var.app_env}"
  data_bucket      = var.data_bucket_name != "" ? var.data_bucket_name : "${var.project_id}-forecasting-data-${var.app_env}"
  rag_bucket       = var.rag_bucket_name != "" ? var.rag_bucket_name : "${var.project_id}-forecasting-rag-${var.app_env}"

  common_labels = {
    env     = var.app_env
    service = "forecasting"
    managed = "terraform"
  }
}

# ---------------------------------------------------------------------------
# Artifacts bucket — model files, feature columns, metadata
# ---------------------------------------------------------------------------

resource "google_storage_bucket" "artifacts" {
  name                        = local.artifacts_bucket
  location                    = var.location
  force_destroy               = var.force_destroy
  uniform_bucket_level_access = true

  versioning {
    enabled = var.artifacts_versioning_enabled
  }

  lifecycle_rule {
    condition {
      num_newer_versions = var.artifacts_keep_versions
      with_state         = "ARCHIVED"
    }
    action {
      type = "Delete"
    }
  }

  # Transition older versions to cheaper storage after 30 days
  lifecycle_rule {
    condition {
      days_since_noncurrent_time = 30
      with_state                 = "ARCHIVED"
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  labels = local.common_labels
}

# ---------------------------------------------------------------------------
# Data bucket — raw CSVs, processed parquet, feature store
# ---------------------------------------------------------------------------

resource "google_storage_bucket" "data" {
  name                        = local.data_bucket
  location                    = var.location
  force_destroy               = var.force_destroy
  uniform_bucket_level_access = true

  lifecycle_rule {
    condition {
      age        = 90
      with_state = "ANY"
      matches_prefix = ["raw/"]
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  labels = local.common_labels
}

# ---------------------------------------------------------------------------
# RAG documents bucket — knowledge base for LLM copilot
# ---------------------------------------------------------------------------

resource "google_storage_bucket" "rag" {
  name                        = local.rag_bucket
  location                    = var.location
  force_destroy               = var.force_destroy
  uniform_bucket_level_access = true

  labels = local.common_labels
}
