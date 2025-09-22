# IIoT ML Anomaly Detection Simulator

A fully-functioning simulation of an Industrial Internet of Things (IIoT) scenario and a binary classification machine learning model for anomaly detection (detecting whether or not an item leaving the assembly line is defective.)

Also contains the dataset in .csv form, as well as the script used to generate the data.

## Features

- Sample data generation
- Stream processing simulator scripts to simulate factory sensors and human quality control staff
- A machine learning model that can be dynamically retrained on new, labelled data, at the push of a button
- A web dashboard that loads the products, their features, the prediction, and live accuracy

## Requirements
- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Installation

This app is dockerized. Simply clone the github repository:
```
git clone https://github.com/edi-tahiri1447/IU-From-Model-to-Production-IoT-Anomaly-Detection
```
and, from inside the project directory, run:
```
docker-compose build
```
to build, and
```
docker-compose up
```
to run.

## Usage

### Accessing the Dashboard/Control Room

Once you have launched the Docker container, please navigate to http://localhost:8000/monitor on your web browser.

### Starting/Stopping Simulators

After accessing the website, all following usage is simple: one presses the buttons labelled for each function.
The procedure for the simulation is intended to be as follows:
1. Reset model, wipe database (if restarting)
2. Start sensor; let predictions accumulate
3. Start inspector; have quality control staff validate predictions; observe accuracy
4. Retrain model; watch the accuracy increase as the model adapts to the new data
All of these steps can, of course, happen concurrently.

### Regenerating/Tweaking Datasets

All the steps made to create the .csv datasets are described in the Jupyter Notebook under data/data_generator.ipynb.