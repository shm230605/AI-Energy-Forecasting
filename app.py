from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("models/energy_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    hour = data["hour"]
    day = data["day"]

    prediction = model.predict([[hour, day]])

    return jsonify({
        "predicted_energy": float(prediction[0])
    })

if __name__ == "__main__":
    app.run(debug=True)