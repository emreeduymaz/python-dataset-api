from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model("outfit_model.h5")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    height = float(data["height"])
    weight = float(data["weight"])
    temperature = float(data["temperature"])
    is_raining = int(data["is_raining"])

    bmi = weight / ((height / 100) ** 2)
    input_data = np.array([[height, weight, bmi, temperature, is_raining]])
    prediction = model.predict(input_data)
    labels = ["heavy coat", "light coat", "light jacket", "sweatshirt", "tshirt"]
    predicted_label = labels[np.argmax(prediction)]

    return jsonify({"prediction": predicted_label})
