terraform {
  required_version = ">= 1.6.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# ---------------------------------------------------------------------------
# Enable required GCP APIs
# ---------------------------------------------------------------------------

resource "google_project_service" "run" {
  service            = "run.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "artifactregistry" {
  service            = "artifactregistry.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "secretmanager" {
  service            = "secretmanager.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "storage" {
  service            = "storage.googleapis.com"
  disable_on_destroy = false
}

# ---------------------------------------------------------------------------
# GCS buckets
# ---------------------------------------------------------------------------

resource "google_storage_bucket" "artifacts" {
  name                        = "${var.project_id}-forecasting-artifacts-${var.app_env}"
  location                    = var.artifacts_bucket_location
  force_destroy               = false
  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      num_newer_versions = 5
    }
    action {
      type = "Delete"
    }
  }

  labels = {
    env     = var.app_env
    service = "forecasting"
  }
}

resource "google_storage_bucket" "data" {
  name                        = "${var.project_id}-forecasting-data-${var.app_env}"
  location                    = var.data_bucket_location
  force_destroy               = false
  uniform_bucket_level_access = true

  labels = {
    env     = var.app_env
    service = "forecasting"
  }
}

# ---------------------------------------------------------------------------
# Service account for the Cloud Run backend
# ---------------------------------------------------------------------------

resource "google_service_account" "backend" {
  account_id   = "forecasting-backend-${var.app_env}"
  display_name = "Forecasting Backend (${var.app_env})"
}

resource "google_storage_bucket_iam_member" "backend_artifacts_reader" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.backend.email}"
}

resource "google_storage_bucket_iam_member" "backend_data_reader" {
  bucket = google_storage_bucket.data.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.backend.email}"
}

resource "google_project_iam_member" "backend_secret_accessor" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.backend.email}"
}

# ---------------------------------------------------------------------------
# Cloud Run — backend service
# ---------------------------------------------------------------------------

resource "google_cloud_run_v2_service" "backend" {
  name     = "forecasting-backend-${var.app_env}"
  location = var.region

  depends_on = [google_project_service.run]

  template {
    service_account = google_service_account.backend.email

    scaling {
      min_instance_count = var.backend_min_instances
      max_instance_count = var.backend_max_instances
    }

    containers {
      image = var.backend_image

      resources {
        limits = {
          cpu    = var.backend_cpu
          memory = var.backend_memory
        }
        startup_cpu_boost = true
      }

      # Application environment
      env {
        name  = "APP_ENV"
        value = var.app_env
      }

      env {
        name  = "GCS_ARTIFACTS_BUCKET"
        value = google_storage_bucket.artifacts.name
      }

      env {
        name  = "GCS_DATA_BUCKET"
        value = google_storage_bucket.data.name
      }

      # OpenAI API key from Secret Manager (optional — skip if secret not configured)
      dynamic "env" {
        for_each = var.openai_secret_version != "" ? [1] : []
        content {
          name = "OPENAI_API_KEY"
          value_source {
            secret_key_ref {
              secret  = split("/versions/", var.openai_secret_version)[0]
              version = split("/versions/", var.openai_secret_version)[1]
            }
          }
        }
      }

      ports {
        container_port = 8000
      }

      startup_probe {
        http_get {
          path = "/health/live"
          port = 8000
        }
        initial_delay_seconds = 10
        period_seconds        = 5
        failure_threshold     = 6
      }

      liveness_probe {
        http_get {
          path = "/health/live"
          port = 8000
        }
        period_seconds    = 30
        failure_threshold = 3
      }
    }

    max_instance_request_concurrency = var.backend_concurrency

    labels = {
      env     = var.app_env
      service = "forecasting-backend"
    }
  }

  labels = {
    env = var.app_env
  }

  lifecycle {
    ignore_changes = [
      # Allow external CI/CD to update the image without Terraform drift
      template[0].containers[0].image,
    ]
  }
}

# Allow unauthenticated access (public API — add IAP or API key auth for production)
resource "google_cloud_run_v2_service_iam_member" "backend_public" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.backend.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ---------------------------------------------------------------------------
# Artifact Registry — Docker repository
# ---------------------------------------------------------------------------

resource "google_artifact_registry_repository" "backend" {
  repository_id = "forecasting-backend"
  location      = var.region
  format        = "DOCKER"
  description   = "Docker images for the forecasting backend service"

  depends_on = [google_project_service.artifactregistry]

  labels = {
    env = var.app_env
  }
}
