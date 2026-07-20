from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)  # Required so the HTML frontend (GitHub Pages) can call this API

model        = joblib.load("superkart_model.pkl")
preprocessor = joblib.load("superkart_preprocessor.pkl")

FEATURE_COLUMNS = [
    "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area",
    "Product_Type", "Product_MRP", "Store_Id", "Store_Establishment_Year",
    "Store_Size", "Store_Location_City_Type", "Store_Type"
]

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "SuperKart Sales Forecasting API", "status": "running"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data    = request.get_json(force=True)
        missing = [f for f in FEATURE_COLUMNS if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        input_df = pd.DataFrame([{col: data[col] for col in FEATURE_COLUMNS}])
        input_df["Product_Sugar_Content"] = (
            input_df["Product_Sugar_Content"].replace("reg", "Regular")
        )
        pred = model.predict(preprocessor.transform(input_df))[0]
        return jsonify({"predicted_sales_total": round(float(pred), 2), "currency": "INR"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
