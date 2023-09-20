from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('svm_model.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data = request.get_json()
    TEMP = request.form['a']
    HUM = request.form['b']
    MOISTURE = request.form['c']
    arr = np.array([[TEMP, HUM, MOISTURE]])
    pred = model.predict(arr)

    response = {'prediction': pred}
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)