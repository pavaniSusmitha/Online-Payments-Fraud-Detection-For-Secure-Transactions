from flask import Flask, render_template, request
import numpy as np
import pickle

model = pickle.load(open('payments.pkl', 'rb'))  # Ensure this file exists

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route("/submit", methods=["POST"])
def submit():
    step = float(request.form['step'])
    type_ = request.form['type']
    amount = float(request.form['amount'])
    oldbalanceOrg = float(request.form['oldbalanceOrg'])
    newbalanceOrig = float(request.form['newbalanceOrig'])
    oldbalanceDest = float(request.form['oldbalanceDest'])
    newbalanceDest = float(request.form['newbalanceDest'])

    # Type encoding: You must match how you trained the model
    type_map = {'CASH_OUT': 0, 'PAYMENT': 1, 'CASH_IN': 2, 'TRANSFER': 3, 'DEBIT': 4}
    type_encoded = type_map.get(type_.upper(), 1)  # default to PAYMENT if unknown

    features = [[step, type_encoded, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]]
    prediction = model.predict(features)

    output = "Fraud" if prediction[0] == 1 else "Not Fraud"
    return render_template("submit.html", result=output)


if __name__ == "__main__":
    app.run(debug=True)
