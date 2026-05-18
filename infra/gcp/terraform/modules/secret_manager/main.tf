locals {
  use_automatic_replication = length(var.secret_replication_locations) == 0
}

# ---------------------------------------------------------------------------
# OpenAI API key
# ---------------------------------------------------------------------------

resource "google_secret_manager_secret" "openai_api_key" {
  count     = var.create_openai_secret ? 1 : 0
  project   = var.project_id
  secret_id = "forecasting-openai-api-key-${var.app_env}"

  replication {
    dynamic "auto" {
      for_each = local.use_automatic_replication ? [1] : []
      content {}
    }

    dynamic "user_managed" {
      for_each = local.use_automatic_replication ? [] : [1]
      content {
        dynamic "replicas" {
          for_each = var.secret_replication_locations
          content {
            location = replicas.value
          }
        }
      }
    }
  }

  labels = {
    env     = var.app_env
    service = "forecasting"
    managed = "terraform"
  }
}

resource "google_secret_manager_secret_version" "openai_api_key" {
  count       = var.create_openai_secret && var.openai_api_key != "" ? 1 : 0
  secret      = google_secret_manager_secret.openai_api_key[0].id
  secret_data = var.openai_api_key

  lifecycle {
    # Prevent accidental destruction of secret versions
    prevent_destroy = false
    ignore_changes  = [secret_data]
  }
}

# ---------------------------------------------------------------------------
# App configuration secret (non-sensitive runtime config)
# ---------------------------------------------------------------------------

resource "google_secret_manager_secret" "app_config" {
  project   = var.project_id
  secret_id = "forecasting-app-config-${var.app_env}"

  replication {
    dynamic "auto" {
      for_each = local.use_automatic_replication ? [1] : []
      content {}
    }

    dynamic "user_managed" {
      for_each = local.use_automatic_replication ? [] : [1]
      content {
        dynamic "replicas" {
          for_each = var.secret_replication_locations
          content {
            location = replicas.value
          }
        }
      }
    }
  }

  labels = {
    env     = var.app_env
    service = "forecasting"
    managed = "terraform"
  }
}
