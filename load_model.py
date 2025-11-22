import joblib
import pandas as pd

# ------------------------------
# 1. Load the trained model
# ------------------------------
MODEL_PATH = "ev_range_rf_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print("\nModel loaded successfully!\n")
except Exception as e:
    print(f"\nError loading model: {e}\n")
    exit()

# ------------------------------
# 2. Take dynamic inputs
# ------------------------------
def take_input():
    print("Enter the EV details to predict range:\n")

    battery_capacity = float(input("Battery Capacity (kWh): "))
    efficiency = float(input("Efficiency (Wh per km): "))
    top_speed = float(input("Top Speed (km/h): "))
    acceleration = float(input("0–100 km/h Acceleration Time (seconds): "))
    torque = float(input("Torque (Nm): "))

    # Extra features required by the model
    usable_battery = battery_capacity * 0.95            # estimated
    performance_index = (top_speed / acceleration) + (torque / 10)

    new_data = pd.DataFrame({
        "battery_capacity_kWh": [battery_capacity],
        "efficiency_wh_per_km": [efficiency],
        "top_speed_kmh": [top_speed],
        "acceleration_0_100_s": [acceleration],
        "torque_nm": [torque],
        "usable_battery_kWh": [usable_battery],
        "performance_index": [performance_index]
    })

    return new_data

# ------------------------------
# 3. Predict using model
# ------------------------------
def predict_range():
    df = take_input()

    try:
        prediction = model.predict(df)
        print("\nPredicted EV Range (km):", round(prediction[0], 2))
    except Exception as e:
        print(f"\nPrediction error: {e}")

# Run prediction
if __name__ == "__main__":
    predict_range()
