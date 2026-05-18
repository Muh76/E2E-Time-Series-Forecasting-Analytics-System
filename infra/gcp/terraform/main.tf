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

  project            = var.project_id
  service            = each.key
  disable_on_destroy = false
}

# ---------------------------------------------------------------------------
# GCS — storage buckets
# ---------------------------------------------------------------------------

module "gcs" {
  source = "./modules/gcs"

  project_id = var.project_id
  app_env    = var.app_env
  location   = var.gcs_location

  depends_on = [google_project_service.apis]
}

# ---------------------------------------------------------------------------
# Artifact Registry — Docker repository
# ---------------------------------------------------------------------------

module "artifact_registry" {
  source = "./modules/artifact_registry"

  project_id = var.project_id
  region     = var.region
  app_env    = var.app_env

  depends_on = [google_project_service.apis]
}

# ---------------------------------------------------------------------------
# IAM — service accounts and role bindings
# ---------------------------------------------------------------------------

module "iam" {
  source = "./modules/iam"

  project_id                      = var.project_id
  app_env                         = var.app_env
  region                          = var.region
  artifacts_bucket_name           = module.gcs.artifacts_bucket_name
  data_bucket_name                = module.gcs.data_bucket_name
  rag_bucket_name                 = module.gcs.rag_bucket_name
  artifact_registry_repository_id = module.artifact_registry.repository_id

  depends_on = [module.gcs, module.artifact_registry]
}

# ---------------------------------------------------------------------------
# Secret Manager — API keys and config
# ---------------------------------------------------------------------------

module "secret_manager" {
  source = "./modules/secret_manager"

  project_id     = var.project_id
  app_env        = var.app_env
  openai_api_key = var.openai_api_key

  depends_on = [google_project_service.apis]
}

# ---------------------------------------------------------------------------
# Cloud Run — FastAPI backend service
# ---------------------------------------------------------------------------

module "cloud_run" {
  source = "./modules/cloud_run"

  project_id            = var.project_id
  region                = var.region
  app_env               = var.app_env
  image                 = var.backend_image
  service_account_email = module.iam.backend_service_account_email
  cpu                   = var.backend_cpu
  memory                = var.backend_memory
  min_instances         = var.backend_min_instances
  max_instances         = var.backend_max_instances
  concurrency           = var.backend_concurrency
  artifacts_bucket_name = module.gcs.artifacts_bucket_name
  data_bucket_name      = module.gcs.data_bucket_name
  rag_bucket_name       = module.gcs.rag_bucket_name

  openai_secret_version_name = module.secret_manager.openai_secret_version_name

  extra_env_vars = {
    LOG_LEVEL = var.app_env == "prod" ? "WARNING" : "INFO"
  }

  depends_on = [module.iam, module.secret_manager]
}

# ---------------------------------------------------------------------------
# Cloud Scheduler — ETL and training jobs
# ---------------------------------------------------------------------------

module "cloud_scheduler" {
  source = "./modules/cloud_scheduler"

  project_id          = var.project_id
  region              = var.region
  app_env             = var.app_env
  scheduler_sa_email  = module.iam.scheduler_service_account_email
  backend_service_url = module.cloud_run.service_url
  etl_schedule        = var.etl_schedule
  training_schedule   = var.training_schedule
  timezone            = var.scheduler_timezone

  depends_on = [module.cloud_run, module.iam]
}
