resource "google_artifact_registry_repository" "backend" {
  project       = var.project_id
  location      = var.region
  repository_id = var.repository_id
  format        = "DOCKER"
  description   = var.description

  cleanup_policies {
    id     = "keep-minimum-versions"
    action = "KEEP"
    most_recent_versions {
      keep_count = var.cleanup_keep_tag_count
    }
  }

  cleanup_policies {
    id     = "delete-untagged"
    action = "DELETE"
    condition {
      tag_state = "UNTAGGED"
      older_than = "168h"  # delete untagged layers older than 7 days
    }
  }

  labels = {
    env     = var.app_env
    service = "forecasting"
    managed = "terraform"
  }
}
