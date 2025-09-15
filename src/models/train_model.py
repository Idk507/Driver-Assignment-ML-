import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import json
from typing import Dict
import pandas as pd
import toml
from pathlib import Path
from lightgbm import LGBMClassifier
import pickle

from src.utils.store import AssignmentStore
from src.models.classifier import SklearnClassifier



os.environ["LOKY_MAX_CPU_COUNT"] = "4"  

def main():
    store = AssignmentStore()
    config = toml.load("config.toml")
    features = config["features"]["features"]
    target = config["target"]["target"]
    train_data = store.get_submission("train_data.csv")
    test_data = store.get_submission("test_data_split.csv")
    target_dist = train_data[target].value_counts(normalize=True).to_dict()
    #LightGBM
    lgb_model = LGBMClassifier(
        random_state=42,
        scale_pos_weight=target_dist[0] / target_dist[1],
        n_estimators=300,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7
    )
    classifier = SklearnClassifier(lgb_model, features, target)
    classifier.train(train_data)

    feature_importance = pd.Series(lgb_model.feature_importances_, index=features)
    metrics = classifier.evaluate(test_data)

    model_path = Path("models/saved_model.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(classifier, f)
    store.put_json(os.path.join(store.submission_dir, "metrics.json"), metrics)
    

if __name__ == "__main__":
    main()