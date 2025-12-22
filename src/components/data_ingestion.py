import os
import pandas as pd
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.logger import logging
from src.exception import CustomException
import sys

class DataIngestion:
    def __init__(self,config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        try:
            logging.info("Starting Data Ingestion")

            if not os.path.exists(self.config.raw_data_path):
                raise FileNotFoundError(
                    f"Raw data not found at {self.config.raw_data_path}"
                )
            df = pd.read_csv(self.config.raw_data_path)
            logging.info(f"Loaded data shape: {df.shape}")

            os.makedirs(self.config.ingestion_dir, exist_ok=True)
            df.to_csv(self.config.ingested_file_path, index=False)

            logging.info(f"Ingested data saved at {self.config.ingested_file_path}")

            return DataIngestionArtifact(
                ingested_file_path= self.config.ingested_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)
