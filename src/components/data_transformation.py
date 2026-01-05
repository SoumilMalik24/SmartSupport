import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact
from src.logger import logging
from src.exception import CustomException
import sys
import pickle


class DataTransformation:

    def __init__(self,config:DataTransformationConfig):
        self.config = config
        self.label_encoder = LabelEncoder()

    
    def initiate_data_transformation(self,raw_data_path:str) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation")

            df = pd.read_csv(raw_data_path)

            df = df[["subject","body","type"]]
            df["text"] = df["subject"].fillna("")+" "+df["body"].fillna("")
            
            df["label"] = self.label_encoder.fit_transform(df["type"])

            train_df, test_df = train_test_split(
                df[["text","label"]],
                test_size= self.config.test_size,
                random_state= self.config.random_state,
                stratify= df['label']
            )

            os.makedirs(os.path.dirname(self.config.transformed_train_path), exist_ok=True)

            train_df.to_csv(self.config.transformed_train_path, index=False)
            test_df.to_csv(self.config.transformed_test_path, index=False)
            

            with open(self.config.tokenizer_path, "wb") as f:
                pickle.dump(self.label_encoder,f)

            logging.info("Data Transformation completed")

            return DataTransformationArtifact(
                train_path=self.config.transformed_train_path,
                test_path=self.config.transformed_test_path,
                tokenizer_path=self.config.tokenizer_path
            )

        except Exception as e:
            raise CustomException(sys,e)