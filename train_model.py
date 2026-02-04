import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier
from data_generator import generate_credit_data

df = generate_credit_data()

X = df.drop("loan_default", axis=1)
y = df["loan_default"]

# Feature Selection
selector = SelectKBest(score_func=f_classif, k=7)
X_selected = selector.fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.1]
}

model = GridSearchCV(
    XGBClassifier(eval_metric="logloss"),
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

with mlflow.start_run():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))

    mlflow.log_params(model.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model.best_estimator_, "model")

joblib.dump((model.best_estimator_, selector), "model.pkl")