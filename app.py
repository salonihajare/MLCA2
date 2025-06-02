from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

model = joblib.load("model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        features = [
            float(data["RAM"]),
            float(data["ROM"]),
            float(data["Mobile_Size"]),
            float(data["Primary_Cam"]),
            float(data["Selfie_Cam"]),
            float(data["Battery_Power"])
        ]
        prediction = model.predict([features])[0]
        return jsonify({"predicted_price": round(prediction, 2)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
