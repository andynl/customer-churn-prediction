from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask('__name__')

model_logistic_regression = joblib.load('../data/output/model-logistic-regression.joblib')
model_random_forest = joblib.load('../data/output/model-random-forest.joblib')
model_gradient_bootsting = joblib.load('../data/output/model-gradient-boosting.joblib')

@app.route('/predict-logistic-regression', methods=['POST'])
def predict_lr():
    data = request.get_json()
    print(data)
    prediction = model_logistic_regression.predict(data)
    return jsonify(round(prediction[0]))

@app.route('/predict-random-forest', methods=['POST'])
def predict_svr():
    data = request.get_json()
    prediction = model_random_forest.predict(data)
    return jsonify(round(prediction[0]))

@app.route('/predict-gradient-boosting', methods=['POST'])
def predict_mlpr():
    data = request.get_json()
    prediction = model_gradient_bootsting.predict(data)
    return jsonify(round(prediction[0]))

if __name__ == '__main__':
    app.run(port=5000, debug=True)