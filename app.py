from flask import Flask, request, jsonify, render_template, redirect
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text

import os
import sys
import subprocess


import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pickle


app = Flask(__name__)

# DB config
app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql://postgres:1234@localhost:5432/iotdb"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define table
class Product(db.Model):
    __tablename__ = "products"

    id = db.Column(db.Integer, primary_key=True)
    temp = db.Column(db.Float, nullable=False)
    humidity = db.Column(db.Float, nullable=False)
    sound_volume = db.Column(db.Float, nullable=False)
    label_predicted = db.Column(db.Integer, nullable=False)
    label_real = db.Column(db.Integer, nullable=True)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())

    true_positive = db.Column(db.Integer, nullable=True)
    false_positive = db.Column(db.Integer, nullable=True)
    true_negative = db.Column(db.Integer, nullable=True)
    false_negative = db.Column(db.Integer, nullable=True)

    accuracy_at_timestamp = db.Column(db.Float, nullable=True)


# Ensure table exists
with app.app_context():
    db.create_all()

# At the start, check if models directory exists; if not, make one
if not os.path.exists(r"./models"):
    os.makedirs(r"./models")
# If models folder is empty: train initial model from init_data.csv, then pickle it
if not os.listdir(r"./models"):
    init_data = pd.read_csv("data/init_data.csv")

    X_train = init_data[["temp", "humidity", "sound_volume"]]
    y_train = init_data["label"]

    clf = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42
        )
    )
    clf.fit(X_train, y_train)

    # Save the whole pipeline
    with open("models/model.pkl", "wb") as f:
        pickle.dump(clf, f)

with open("models/model.pkl", "rb") as f:
    clf = pickle.load(f)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = np.array([[data["temp"], data["humidity"], data["sound_volume"]]])
    prediction = clf.predict(features)[0]

    print(f"Received features: temp: {data["temp"]}, humidity: {data["humidity"]}, sound volume: {data["sound_volume"]}.")
    print(f"Predicted: {prediction}")

    prediction_in_db = Product(
        id=data["id"],
        temp=data["temp"],
        humidity=data["humidity"],
        sound_volume=data["sound_volume"],
        label_predicted=int(prediction)
    )

    db.session.add(prediction_in_db)
    db.session.commit()

    return jsonify({"input": data, "prediction": int(prediction)}) # returns inputs and outputs of prediction


@app.route("/label", methods=["POST"])
def label():
    data = request.get_json()
    label = data["label"]
    id = data["id"]

    product = Product.query.get(id)
    if product is None:
        return jsonify({f"error: Product {id} not found"}), 404

    product.label_real = label

    product.true_positive = int(product.label_predicted == 1 and product.label_real == 1)
    product.false_positive = int(product.label_predicted == 1 and product.label_real == 0)
    product.true_negative = int(product.label_predicted == 0 and product.label_real == 0)
    product.false_negative = int(product.label_predicted == 0 and product.label_real == 1)

    # get all predictions up to current timestamp
    rows = Product.query.filter(Product.timestamp <= product.timestamp).all()

    tp = sum((r.true_positive or 0) for r in rows)
    fp = sum((r.false_positive or 0) for r in rows)
    tn = sum((r.true_negative or 0) for r in rows)
    fn = sum((r.false_negative or 0) for r in rows)

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else None
    product.accuracy_at_timestamp = accuracy

    if product.label_predicted == product.label_real:
        print(f"Prediction made for product {id} was CORRECT.")
    else:
        print(f"Prediction made for product {id} was INCORRECT.")
    print(f"Accuracy is now at {accuracy}.")

    db.session.commit()

    return jsonify({"id": id, "label_real": label})


@app.route("/next_unpredicted", methods=["GET"])
def next_unpredicted():
    product = Product.query.order_by(Product.id.desc()).first()  # highest ID so far
    if product:
        return {"id": product.id + 1}  # next available ID
    else:
        return {"id": 1}  # database is empty, start from 1
    

@app.route("/next_unlabeled", methods=["GET"])
def next_unlabeled():
    product = Product.query.filter_by(label_real=None).order_by(Product.id.asc()).first()
    if product:
        return {"id": product.id}
    else:
        return {"id": 1}
    

# Creating endpoints to start/stop sensor_sim.py and inspector_sim.py (with buttons) for easy simulation
sensor_process = None
inspector_process = None


@app.route("/start_sensor", methods=["POST"])
def start_sensor():
    global sensor_process
    if sensor_process is None or sensor_process.poll() is not None:
        sensor_process = subprocess.Popen([sys.executable, "sensor_sim.py"])

        return {"status": "Sensor simulator started."}
    else:
        return {"status": "Sensor simulator is already running."}


@app.route("/stop_sensor", methods=["POST"])
def stop_sensor():
    global sensor_process
    if sensor_process is not None and sensor_process.poll() is None:
        sensor_process.terminate()
        sensor_process = None

        return {"status": "Sensor simulator stopped."}
    else:
        return {"status": "Sensor simulator is not running."}


@app.route("/start_inspector", methods=["POST"])
def start_inspector():
    global inspector_process

    if inspector_process is None or inspector_process.poll() is not None:
        inspector_process = subprocess.Popen([sys.executable, "inspector_sim.py"])

        return {"status": "Inspector simulator started."}
    else:
        return {"status": "Inspector simulator is already running."}


@app.route("/stop_inspector", methods=["POST"])
def stop_inspector():
    global inspector_process

    if inspector_process is not None and inspector_process.poll() is None:
        inspector_process.terminate()
        inspector_process = None

        return {"status": "Inspector simulator stopped."}
    else:
        return {"status": "Inspector simulator is not running."}


@app.route("/retrain_model", methods=["POST"])
def retrain_model():
    try:
        # Get all rows with both predicted and real labels
        rows = Product.query.filter(
            Product.label_real.isnot(None)
        ).all()

        if not rows:
            return {"status": "No labeled data available for retraining."}, 400

        # Extract features and labels
        X_train = np.array([[r.temp, r.humidity, r.sound_volume] for r in rows])
        y_train = np.array([r.label_real for r in rows])

        # Train a new pipeline
        new_clf = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42
            )
        )
        new_clf.fit(X_train, y_train)

        # Save the updated model
        with open("models/model.pkl", "wb") as f:
            pickle.dump(new_clf, f)

        # Update the global clf in memory
        global clf
        clf = new_clf

        return {"status": f"Model retrained on {len(rows)} labeled rows."}

    except Exception as e:
        return {"status": f"Error retraining model: {e}"}, 500
    

@app.route("/reset_model", methods=["POST"])
def reset_model():
    try:
        # Reload init data
        init_data = pd.read_csv("data/init_data.csv")

        X_train = init_data[["temp", "humidity", "sound_volume"]]
        y_train = init_data["label"]

        # Train a new pipeline
        new_clf = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42
            )
        )
        new_clf.fit(X_train, y_train)

        # Save the updated model
        with open("models/model.pkl", "wb") as f:
            pickle.dump(new_clf, f)

        new_clf.fit(X_train, y_train)

        # Update the global clf in memory
        global clf
        clf = new_clf

        return {"status": "Model reset to model trained on the initial data."}
    except Exception as e:
        return {"status": f"Error resetting model: {e}"}, 500


@app.route("/wipe_database", methods=["POST"])
def wipe_database():
    try:
        db.session.execute(text("TRUNCATE TABLE products RESTART IDENTITY CASCADE"))
        db.session.commit()

        return {"status": "Database wiped."}
    except Exception as e:
        db.session.rollback()
        return {"status": f"Error wiping database: {e}"}, 500


@app.route("/monitor")
def dashboard():
    # Grab last 20 predictions for display
    products = Product.query.order_by(Product.timestamp.desc()).limit(25).all()
    return render_template("dashboard.html", products=products)
    

@app.route("/latest_predictions", methods=["GET"])
def latest_predictions():
    rows = Product.query.order_by(Product.id.desc()).all()
    data = []
    for r in rows:
        data.append({
            "id": r.id,
            "temp": r.temp,
            "humidity": r.humidity,
            "sound_volume": r.sound_volume,
            "label_predicted": r.label_predicted,
            "label_real": r.label_real if r.label_real is not None else "-",
            "accuracy_at_timestamp": r.accuracy_at_timestamp,
            "timestamp": str(r.timestamp)
        })
    return {"products": data}


@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.ico')


if __name__ == "__main__":
    import webbrowser
    import threading

    # Open the dashboard in the default browser after a short delay
    def open_browser():
        webbrowser.open_new("http://127.0.0.1:5000/monitor")

    threading.Timer(1.0, open_browser).start()  # 1 second delay
    app.run(debug=True, host="0.0.0.0", port=5000)