import time
import requests

import pandas as pd

# Config
CSV_FILE = "data/new_data.csv"
LABEL_ENDPOINT = "http://127.0.0.1:5000/label"
NEXT_ID_ENDPOINT = "http://127.0.0.1:5000/next_unlabeled"
DELAY = 0.5

# Load dataset
dataset = pd.read_csv(CSV_FILE)

# Stagger it from sensor_sim.py; assume both are run at once
time.sleep(DELAY)

while True:
    # Ask server for next unlabelled ID
    label_resp = requests.get(NEXT_ID_ENDPOINT)
    next_id_info = label_resp.json()
    next_id = next_id_info.get("id")

    print(f"Starting at next_id: {next_id}")
    
    if next_id is None:
        print("No more work left")
        break

    # Fetch the row from local CSV or DB and send label
    row = dataset.loc[next_id, :]
    payload = {"id": next_id, "label": row["label"]}
    requests.post(LABEL_ENDPOINT, json=payload)

    time.sleep(DELAY)