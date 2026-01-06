from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.constant.training_pipeline import *

from src.logger import logging
import os


class TrainingPipeline:

    def start_data_ingestion(self):
        logging.info("Pipeline: Data Ingestion started")
        config = DataIngestionConfig()
        ingestion = DataIngestion(config)
        artifact = ingestion.initiate_data_ingestion()
        logging.info("Pipeline: Data Ingestion completed")
        return artifact

    def start_data_validation(self, ingestion_artifact):
        logging.info("Pipeline: Data Validation started")
        config = DataValidationConfig()
        validation = DataValidation(config)
        artifact = validation.initiate_data_validation(
            ingestion_artifact.ingested_file_path
        )
        logging.info("Pipeline: Data Validation completed")
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

    def start_model_trainer(self, transformation_artifact):
        logging.info("Pipeline: Model Trainer started")

        trainer_config = ModelTrainerConfig(
            model_name=MODEL_NAME,
            epochs=MODEL_EPOCHS,
            batch_size=MODEL_BATCH_SIZE,
            learning_rate=MODEL_LEARNING_RATE,
            model_output_dir=MODEL_OUTPUT_DIR
        )

        trainer = ModelTrainer(trainer_config)

        trainer_artifact = trainer.initiate_model_trainer(
            train_path=transformation_artifact.train_path,
            test_path=transformation_artifact.test_path
        )

        logging.info("Pipeline: Model Trainer completed")
        return trainer_artifact

    def run_pipeline(self):
        logging.info("Training Pipeline started")

        ingestion_artifact = self.start_data_ingestion()
        self.start_data_validation(ingestion_artifact)
        transformation_artifact = self.start_data_transformation(ingestion_artifact)
        trainer_artifact = self.start_model_trainer(transformation_artifact)

        logging.info("Training Pipeline completed successfully")
        return trainer_artifact
