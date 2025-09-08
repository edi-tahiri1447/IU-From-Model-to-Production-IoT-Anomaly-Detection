import time
import requests
import pandas as pd

# Config
CSV_FILE = "data/new_data.csv"
PREDICT_ENDPOINT = "http://127.0.0.1:5000/predict"
NEXT_ID_ENDPOINT = "http://127.0.0.1:5000/next_unpredicted"
DELAY = 0.5

# Load dataset
dataset = pd.read_csv(CSV_FILE)

# Ask app where to start predicting
resp = requests.get(NEXT_ID_ENDPOINT)
next_id_info = resp.json()
start_id = next_id_info.get("id", 1)
print(f"Start ID (index) to begin predicting as received from get request is: {start_id}")

# Only iterate from the "starting" id
for _, row in dataset.loc[start_id:, :].iterrows():
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

    time.sleep(DELAY)
