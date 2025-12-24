from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig
)
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.logger import logging

class TrainingPipeline:
    def start_data_ingestion(self):
        logging.info("Pipeline Data Ingestion started")
        config = DataIngestionConfig()
        ingestion = DataIngestion(config)
        artifact = ingestion.initiate_data_ingestion()
        logging.info("Pipeline: Data Ingestion Completed")
        return artifact

    def start_data_validation(self, ingestion_artifact):
        logging.info("Pipeline: Data Validation started")
        config = DataValidationConfig()
        validation = DataValidation(config)
        artifact = validation.initiate_data_validation(
            ingestion_artifact.ingested_file_path
        )
        return artifact