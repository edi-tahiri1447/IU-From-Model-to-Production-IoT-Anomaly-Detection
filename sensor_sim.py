import time
import requests
import pandas as pd

# Config
CSV_FILE = "data/new_data.csv"
PREDICT_ENDPOINT = "http://iot_dashboard:8000/predict"
NEXT_ID_ENDPOINT = "http://iot_dashboard:8000/next_unpredicted"
STATUS_ENDPOINT = "http://iot_dashboard:8000/status_sensor"

# Load dataset
dataset = pd.read_csv(CSV_FILE)

while True:
    try:
        resp = requests.get(STATUS_ENDPOINT)
        status = resp.json().get("status")

        if status == "on":
            # Ask app where to start predicting
            resp = requests.get(NEXT_ID_ENDPOINT)
            next_id_info = resp.json()
            next_id = next_id_info.get("id", 1)

            row = dataset[dataset["id"] == next_id].iloc[0, :]
            payload = {
                "id": row["id"],
                "temp": row["temp"],
                "humidity": row["humidity"],
                "sound_volume": row["sound_volume"]
            }
            
            try:
                predict_resp = requests.post(PREDICT_ENDPOINT, json=payload)
                predict_result = predict_resp.json()
            except Exception as e:
                print("Error sending data:", e)
        else:
            time.sleep(1)
    except Exception as e:
        print(f"Error talking to app: {e}")
    time.sleep(1)
