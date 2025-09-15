import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score

class Classifier(ABC):
    @abstractmethod
    def train(self, *params) -> None:
        pass

    @abstractmethod
    def evaluate(self, *params) -> Dict[str, float]:
        pass

    @abstractmethod
    def predict(self, *params) -> np.ndarray:
        pass

class SklearnClassifier(Classifier):
    def __init__(self, estimator: BaseEstimator, features: List[str], target: str):
        self.clf = estimator
        self.features = features
        self.target = target

    def train(self, df_train: pd.DataFrame):
        self.clf.fit(df_train[self.features], df_train[self.target])

    def evaluate(self, df_test: pd.DataFrame) -> Dict[str, float]:
        y_true = df_test[self.target].values
        y_pred_proba = self.predict(df_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "auc_roc": roc_auc_score(y_true, y_pred_proba),
            "recall": recall_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred)
        }
        return metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(df[self.features])[:, 1]