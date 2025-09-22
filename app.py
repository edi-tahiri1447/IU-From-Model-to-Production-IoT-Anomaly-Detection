from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text

import os

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pickle

app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/iot_db"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

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


class ServiceControl(db.Model):
    __tablename__ = "service_control"
    name = db.Column(db.String, primary_key=True)
    running = db.Column(db.Boolean, default=False)


def init_service_control():
    for name in ["sensor", "inspector"]:
        svc = ServiceControl.query.filter_by(name=name).first()
        if not svc:
            svc = ServiceControl(name=name, running=False)
            db.session.add(svc)
    db.session.commit()


def init_model():
    # At the start, check if models directory exists; if not, make one
    if not os.path.exists(r"./models"):
        os.makedirs(r"./models")

    # If models folder is empty: train initial model from init_data.csv, then pickle it
    if not os.listdir(r"./models"):
        init_data = pd.read_csv("data/init_data.csv")

        X_train = init_data[["temp", "humidity", "sound_volume"]].to_numpy()
        y_train = init_data["label"].to_numpy()

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
    return clf


def setup_app():
    with app.app_context():
        db.create_all()
        init_service_control()
        
        global clf
        clf = init_model()

setup_app()


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = np.array([[data["temp"], data["humidity"], data["sound_volume"]]])
    prediction = clf.predict(features)[0]

    print(f"Received features: temp: {data['temp']}, humidity: {data['humidity']}, sound volume: {data['sound_volume']}.")
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
    

@app.route("/status_sensor", methods=["GET"])
def status_sensor():
    sensor = db.session.get(ServiceControl, "sensor")
    status = "on" if sensor.running else "off"
    return jsonify({"status": status})


@app.route("/status_inspector", methods=["GET"])
def status_inspector():
    inspector = db.session.get(ServiceControl, "inspector")
    status = "on" if inspector.running else "off"
    return jsonify({"status": status})


@app.route("/start_sensor", methods=["POST"])
def start_sensor():
    sensor = db.session.get(ServiceControl, "sensor")
    if sensor:
        sensor.running = True
        db.session.commit()
        return jsonify({"status": "Sensor start signal sent."})
    return jsonify({"status": "Sensor not found"}), 404


@app.route("/stop_sensor", methods=["POST"])
def stop_sensor():
    sensor = db.session.get(ServiceControl, "sensor")
    if sensor:
        sensor.running = False
        db.session.commit()
        return jsonify({"status": "Sensor stop signal sent."})
    return jsonify({"status": "Sensor not found"}), 404


@app.route("/start_inspector", methods=["POST"])
def start_inspector():
    inspector = db.session.get(ServiceControl, "inspector")
    if inspector:
        inspector.running = True
        db.session.commit()
        return jsonify({"status": "Inspector start signal sent."})
    return jsonify({"status": "Inspector not found"}), 404


@app.route("/stop_inspector", methods=["POST"])
def stop_inspector():
    inspector = db.session.get(ServiceControl, "inspector")
    if inspector:
        inspector.running = False
        db.session.commit()
        return jsonify({"status": "Inspector stop signal sent."})
    return jsonify({"status": "Inspector not found"}), 404


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
    app.run(debug=True, host="0.0.0.0", port=8000)