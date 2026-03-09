import joblib
import numpy as np


model = joblib.load("smart_irrigation_xgboost.pkl")
label_encoder = joblib.load("label_encoder.pkl")


temperature = 29.4      
pressure = 9983.11       
altitude = 12.33     
soil_moisture = 188     

input_data = np.array([[temperature, pressure, altitude, soil_moisture]])

predicted_class_index = model.predict(input_data)[0]
predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]

if predicted_class in ["Very Dry", "Dry"]:
    irrigation_status = "PUMP ON"
else:
    irrigation_status = "PUMP OFF"

print(" Temperature:", temperature)
print(" Pressure:", pressure)
print(" Altitude:", altitude)
print(" Soil Moisture:", soil_moisture)

print("\n Predicted Soil Condition:", predicted_class)
print(" Irrigation Decision:", irrigation_status)
