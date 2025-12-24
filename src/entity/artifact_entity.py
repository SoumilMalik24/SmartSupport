from dataclasses import dataclass

@dataclass 
class DataIngestionArtifact:
    ingested_file_path: str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    message: str