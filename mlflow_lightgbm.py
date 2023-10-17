import logging
import sys
import warnings
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split
from imblearn.pipeline import make_pipeline
from lightgbm import LGBMClassifier
from imblearn.over_sampling import RandomOverSampler
from dotenv import load_dotenv
import os
import joblib

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
load_dotenv()


datafile = os.environ.get("FILE_PATH")
targetfile = os.environ.get("TARGET_FILE_PATH")

def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    recall = recall_score(actual, pred)
    precision = precision_score(actual, pred)
    f1 = f1_score(actual, pred)
    f2_measure = fbeta_score(actual, pred, beta=2)
    return precision, recall, f1, accuracy, f2_measure


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Read the wine-quality csv file from the URL
    data = joblib.load(datafile)
    target = pd.read_csv(targetfile)

    # Split the data into training and test sets. (0.75, 0.25) split.
    X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['SK_ID_CURR']), target['TARGET'], random_state=42)


    with mlflow.start_run():
        ros = RandomOverSampler(sampling_strategy='auto',
                                    random_state=101)
        gbm = LGBMClassifier(n_estimators=200, boosting_type='goss')
        #pipe = Pipeline([('Oversampler', ros), ('light_gbm', gbm)])
        pipe = make_pipeline(ros, gbm)
        pipe.fit(X_train, y_train)
        predicted_targets = pipe.predict(X_test)

        precision, recall, f1, accuracy, f2 = eval_metrics(y_test, predicted_targets)


        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("boosting_type", "goss")
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f2", f2)

        predictions = pipe.predict(X_train)
        signature = infer_signature(X_train, predictions)

        mlflow.sklearn.log_model(pipe, "model", signature=signature)
