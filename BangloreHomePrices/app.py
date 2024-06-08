from flask import Flask, request, redirect, url_for, render_template
import pickle
import numpy as np
import json
import os

app = Flask(__name__)

# Define the base directory
base_dir = os.path.abspath(os.path.dirname(__file__))

# Load the model
model = pickle.load(open(os.path.join(base_dir, "model", "banglore_home_prices_model.pickle"), "rb"))

# Load the columns
with open(os.path.join(base_dir, "static", "columns.json")) as f:
    data_columns = json.load(f)['data_columns']
    locations = data_columns[3:]  # first 3 columns are sqft, bath, and bhk


@app.route("/")
def load_home():
    return render_template("index.html")


@app.route("/result/<float:score>")
def get_result(score):
    return render_template("result.html", score=score)


@app.route("/predict", methods=['POST'])
def predict_home_price():
    try:
        total_sqft = float(request.form["area"])
        bhk = int(request.form["bhk"])
        bath = int(request.form["bath"])
        location = str(request.form["location"])

        # Prepare the input data
        loc_index = data_columns.index(location.lower()) if location.lower() in data_columns else -1

        x = np.zeros(len(data_columns))
        x[0] = total_sqft
        x[1] = bath
        x[2] = bhk
        if loc_index >= 0:
            x[loc_index] = 1

        result = model.predict([x])[0]
        return redirect(url_for("get_result", score=round(result, 2)))
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
