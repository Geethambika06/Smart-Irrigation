from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("smart_irrigation_xgboost.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    irrigation_status = None
    input_data = None

    if request.method == "POST":
        temperature = float(request.form["temperature"])
        pressure = float(request.form["pressure"])
        altitude = float(request.form["altitude"])
        soil_moisture = float(request.form["soil_moisture"])

        input_data = [temperature, pressure, altitude, soil_moisture]

        X = np.array([input_data])
        pred_index = model.predict(X)[0]
        prediction = label_encoder.inverse_transform([pred_index])[0]

        irrigation_status = "PUMP ON" if prediction in ["Very Dry", "Dry"] else "PUMP OFF"

    return render_template(
        "index.html",
        prediction=prediction,
        irrigation_status=irrigation_status,
        input_data=input_data
    )

if __name__ == "__main__":
    app.run(debug=True)
