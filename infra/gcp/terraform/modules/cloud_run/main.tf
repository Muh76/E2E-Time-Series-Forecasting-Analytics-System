locals {
  service_name = var.service_name != "" ? var.service_name : "forecasting-backend-${var.app_env}"
}

resource "google_cloud_run_v2_service" "backend" {
  project  = var.project_id
  name     = local.service_name
  location = var.region

  template {
    service_account = var.service_account_email
    timeout         = "${var.timeout_seconds}s"

    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    max_instance_request_concurrency = var.concurrency

    containers {
      image = var.image

      resources {
        limits = {
          cpu    = var.cpu
          memory = var.memory
        }
        cpu_idle          = var.min_instances == 0
        startup_cpu_boost = true
      }

      # -----------------------------------------------------------------------
      # Plain-text environment variables
      # -----------------------------------------------------------------------

      env {
        name  = "APP_ENV"
        value = var.app_env
      }

      env {
        name  = "GCS_ARTIFACTS_BUCKET"
        value = var.artifacts_bucket_name
      }

      env {
        name  = "GCS_DATA_BUCKET"
        value = var.data_bucket_name
      }

      env {
        name  = "GCS_RAG_BUCKET"
        value = var.rag_bucket_name
      }

      dynamic "env" {
        for_each = var.extra_env_vars
        content {
          name  = env.key
          value = env.value
        }
      }

      # -----------------------------------------------------------------------
      # Secret-backed environment variables
      # -----------------------------------------------------------------------

      dynamic "env" {
        for_each = var.openai_secret_version_name != "" ? [1] : []
        content {
          name = "OPENAI_API_KEY"
          value_source {
            secret_key_ref {
              secret  = split("/versions/", var.openai_secret_version_name)[0]
              version = element(split("/versions/", var.openai_secret_version_name), 1)
            }
          }
        }
      }

      # -----------------------------------------------------------------------
      # Port and health checks
      # -----------------------------------------------------------------------

      ports {
        name           = "http1"
        container_port = 8000
      }

      startup_probe {
        http_get {
          path = "/health/live"
          port = 8000
        }
        initial_delay_seconds = 10
        period_seconds        = 5
        failure_threshold     = 12
        timeout_seconds       = 3
      }

      liveness_probe {
        http_get {
          path = "/health/live"
          port = 8000
        }
        period_seconds    = 30
        failure_threshold = 3
        timeout_seconds   = 5
      }
    }

    labels = {
      env     = var.app_env
      service = "forecasting-backend"
      managed = "terraform"
    }
  }

  labels = {
    env     = var.app_env
    managed = "terraform"
  }

  lifecycle {
    ignore_changes = [
      # CI/CD deploys new image tags; prevent Terraform drift on image URI
      template[0].containers[0].image,
    ]
  }
}

# ---------------------------------------------------------------------------
# IAM — optional public access
# ---------------------------------------------------------------------------

resource "google_cloud_run_v2_service_iam_member" "public_invoker" {
  count    = var.allow_unauthenticated ? 1 : 0
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.backend.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
