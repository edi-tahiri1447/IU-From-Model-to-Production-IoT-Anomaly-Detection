import time
import requests

import pandas as pd

# Config
CSV_FILE = "data/new_data.csv"
LABEL_ENDPOINT = "http://iot_dashboard:8000/label"
NEXT_ID_ENDPOINT = "http://iot_dashboard:8000/next_unlabeled"
STATUS_ENDPOINT = "http://iot_dashboard:8000/status_inspector"

# Load dataset
dataset = pd.read_csv(CSV_FILE)

while True:
    try:
        resp = requests.get(STATUS_ENDPOINT)
        status = resp.json().get("status")

        if status == "on":
            # Ask server for next unlabelled ID
            label_resp = requests.get(NEXT_ID_ENDPOINT)
            next_id_info = label_resp.json()
            next_id = next_id_info.get("id")

            # Fetch the row from local CSV or DB and send label
            row = dataset.loc[next_id, :]
            payload = {"id": next_id, "label": row["label"]}
            requests.post(LABEL_ENDPOINT, json=payload)
        else:
            pass
    except Exception as e:
        print(f"Error talking to app: {e}")
    time.sleep(1)


    