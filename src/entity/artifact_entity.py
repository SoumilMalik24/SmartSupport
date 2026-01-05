from dataclasses import dataclass

@dataclass 
class DataIngestionArtifact:
    ingested_file_path: str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    message: str

@dataclass
class DataTransformationArtifact:
    train_path: str
    test_path: str
    tokenizer_path: str