import os
import sys
import yaml
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataValidationArtifact

class DataValidation:
    def __init__(self,config: DataValidationConfig):
        self.config = config

    def _read_schema(self):
        with open(self.config.schema_file_path, "r") as f:
            return yaml.safe_load(f)

    def initiate_data_validation(self,data_path: str) -> DataValidationArtifact:
        try:
            logging.info("Starting Data Validation")

            df = pd.read_csv(data_path)
            schema = self._read_schema()
        
            # Check empty dataset
            if df.shape[0] == 0:
                return DataValidationArtifact(False, "Dataset is empty")

            # Chech required columns
            missing_cols = [
                col for col in schema["required_columns"]
                if col not in df.columns
            ]
            if missing_cols:
                return DataValidationArtifact(
                    False, f"Missing required columns: {missing_cols}"
                )

            # Check nulls
            if not schema["checks"]["allow_nulls"]:
                if df.isnull().sum().any():
                    return DataValidationArtifact(False, "Null values found in dataset")

                
            logging.info("Data validation successful")
            return DataValidationArtifact(True, "Validation passed")

        except Exception as e:
            raise CustomException(e, sys)
            
