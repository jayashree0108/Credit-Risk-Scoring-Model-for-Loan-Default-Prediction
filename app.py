from flask import Flask, request, jsonify
from src.predict import predict_risk

app = Flask(__name__)

@app.route("/")
def home():
    return "Credit Risk Scoring API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    risk = predict_risk(data)
    return jsonify({"default_risk_probability": float(risk)})

if __name__ == "__main__":
    app.run(debug=True)