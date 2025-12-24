from dataclasses import dataclass
from src.constant.application import(
    RAW_DATA_DIR,
    RAW_DATA_FILE,
    ARTIFACT_DIR,
    DATA_INGESTION_DIR,
    INGESTED_DATA_FILE
)
import os

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join(RAW_DATA_DIR, RAW_DATA_FILE)
    artifact_dir: str = ARTIFACT_DIR
    ingestion_dir: str = os.path.join(ARTIFACT_DIR,DATA_INGESTION_DIR)
    ingested_file_path:str = os.path.join(ARTIFACT_DIR,DATA_INGESTION_DIR,INGESTED_DATA_FILE)


@dataclass
class DataValidationConfig:
    schema_file_path: str = os.path.join("config","schema.yaml")
    validation_dir: str = os.path.join(ARTIFACT_DIR, "data_validation")