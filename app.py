from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('svm_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET'])
def predict_from_url():
    # Extract query parameters from the URL
    temp = request.args.get('temp')
    hum = request.args.get('hum')
    moisture = request.args.get('moisture')

    if temp is not None and hum is not None and moisture is not None:
        try:
            # Convert to appropriate data types if needed
            temp = float(temp)
            hum = float(hum)
            moisture = float(moisture)

            # Perform prediction based on extracted values
            arr = np.array([[temp, hum, moisture]])
            pred = model.predict(arr)

            # You can return the prediction result as a response
            return f'Prediction Result: {pred}'
        except ValueError:
            return 'Invalid input data types'
    else:
        return 'Missing input parameters'

if __name__ == "__main__":
    app.run(debug=True)
