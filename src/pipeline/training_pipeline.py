from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig
)

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation

from src.constant.training_pipeline import (
    DATA_TRANSFORMATION_DIR,
    TRANSFORMED_TRAIN_FILE,
    TRANSFORMED_TEST_FILE,
    LABEL_ENCODER_FILE
)

from src.logger import logging
import os

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

    def start_data_transformation(self, ingestion_artifact):
        logging.info("Pipeline: Data Transformation started")

        config = DataTransformationConfig(
            transformed_train_path=os.path.join(
                DATA_TRANSFORMATION_DIR, TRANSFORMED_TRAIN_FILE
            ),
            transformed_test_path=os.path.join(
                DATA_TRANSFORMATION_DIR, TRANSFORMED_TEST_FILE
            ),
            tokenizer_path=os.path.join(
                DATA_TRANSFORMATION_DIR, LABEL_ENCODER_FILE
            )
        )

        transformation = DataTransformation(config)

        artifact = transformation.initiate_data_transformation(
            raw_data_path=ingestion_artifact.ingested_file_path
        )

        logging.info("Pipeline: Data Transformation completed")
        return artifact


    def run_pipeline(self):
        logging.info("Training Pipeline started")

        ingestion_artifact = self.start_data_ingestion()
        validation_artifact = self.start_data_validation(ingestion_artifact)
        transformation_artifact = self.start_data_transformation(ingestion_artifact)

        logging.info("Training Pipeline completed")
        return transformation_artifact
