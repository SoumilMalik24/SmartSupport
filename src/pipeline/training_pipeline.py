from src.entity.config_entity import DataIngestionConfig
from src.components.data_ingestion import DataIngestion
from src.logger import logging

class TrainingPipeline:
    def start_data_ingestion(self):
        logging.info("Pipeline Data Ingestion started")
        config = DataIngestionConfig()
        ingestion = DataIngestion(config)
        artifact = ingestion.initiate_data_ingestion()
        logging.info("Pipeline: Data Ingestion Completed")
        return artifact