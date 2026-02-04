import joblib
import numpy as np

model, selector = joblib.load("model.pkl")

def predict_risk(input_data: dict):
    features = np.array([list(input_data.values())])
    features_selected = selector.transform(features)
    prob = model.predict_proba(features_selected)[0][1]
    return prob