# import mlflow

# mlflow.set_experiment("smartsupport-baseline-test")

# with mlflow.start_run(run_name="mlflow_baseline_check"):
#     mlflow.log_param("stage","setup")
#     mlflow.log_metric("ping", 1.0)



# from src.pipeline.training_pipeline import TrainingPipeline

# if __name__ == "__main__":
#     pipeline = TrainingPipeline()
#     artifact = pipeline.start_data_ingestion()
#     print(artifact)

from src.pipeline.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    pipeline = TrainingPipeline()

    ingestion_artifact = pipeline.start_data_ingestion()
    validation_artifact = pipeline.start_data_validation(ingestion_artifact)

    print(validation_artifact)
