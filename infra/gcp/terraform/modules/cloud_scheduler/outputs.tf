output "etl_job_name" {
  description = "Name of the daily ETL Cloud Scheduler job"
  value       = google_cloud_scheduler_job.etl_daily.name
}

output "etl_job_schedule" {
  description = "Cron schedule of the ETL job"
  value       = google_cloud_scheduler_job.etl_daily.schedule
}

output "training_job_name" {
  description = "Name of the weekly training Cloud Scheduler job"
  value       = google_cloud_scheduler_job.training_weekly.name
}

output "training_job_schedule" {
  description = "Cron schedule of the training job"
  value       = google_cloud_scheduler_job.training_weekly.schedule
}
