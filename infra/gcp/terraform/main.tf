# ---------------------------------------------------------------------------
# Enable required GCP APIs
# ---------------------------------------------------------------------------

resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "secretmanager.googleapis.com",
    "storage.googleapis.com",
    "cloudscheduler.googleapis.com",
    "cloudbuild.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
  ])

  service            = each.key
  disable_on_destroy = false
}
