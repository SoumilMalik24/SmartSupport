import os
import sys
import numpy as np
import pandas as pd

from datasets import Dataset
from transformers import(
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from sklearn.metrics import accuracy_score, f1_score

import mlflow
import mlflow.pytorch

from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import ModelTrainerArtifact
from src.logger import logging
from src.exception import CustomException
import dagshub
dagshub.init(
    repo_owner="SoumilMalik24",
    repo_name="SmartSupport",
    mlflow=True
)


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def _compute_metrics(self,eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return{
            "accuracy": accuracy_score(labels,preds),
            "macro_f1": f1_score(labels,preds, average="macro")
        }
    
    def initiate_model_trainer(self,train_path: str, test_path: str) -> ModelTrainerArtifact:
        try:
            logging.info("Starting Model Trainer")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            num_labels = train_df["label"].nunique()

            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

            def tokenize(batch):
                return tokenizer(
                    batch['text'],
                    truncation = True,
                    padding = "max_length",
                    max_length=128
                )
            
            
            train_ds = Dataset.from_pandas(train_df).map(tokenize, batched=True)
            test_ds = Dataset.from_pandas(test_df).map(tokenize, batched=True)

            train_ds = train_ds.remove_columns(["text"])
            test_ds = test_ds.remove_columns(["text"])
            train_ds.set_format("torch")
            test_ds.set_format("torch")

            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=num_labels
            )

            args = TrainingArguments(
                output_dir=self.config.model_output_dir,
                eval_strategy="epoch",
                save_strategy="no",
                learning_rate=self.config.learning_rate,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                num_train_epochs=self.config.epochs,
                logging_steps=50,
                report_to="none"
            )

            with mlflow.start_run(run_name="transformer_production"):

                mlflow.log_params({
                    "model_name": self.config.model_name,
                    "epochs": self.config.epochs,
                    "batch_size": self.config.batch_size,
                    "learning_rate": self.config.learning_rate
                })

                trainer = Trainer(
                    model=model,
                    args=args,
                    train_dataset=train_ds,
                    eval_dataset=test_ds,
                    compute_metrics=self._compute_metrics
                )

                trainer.train()
                metrics = trainer.evaluate()

                mlflow.log_metric("accuracy", metrics["eval_accuracy"])
                mlflow.log_metric("macro_f1", metrics["eval_macro_f1"])

                model_path = os.path.join(self.config.model_output_dir, "model")
                mlflow.pytorch.save_model(model, model_path)

            logging.info("Model training completed")

            return ModelTrainerArtifact(
                model_path=model_path,
                accuracy=metrics["eval_accuracy"],
                macro_f1=metrics["eval_macro_f1"]
            )

        except Exception as e:
            raise CustomException(e, sys)