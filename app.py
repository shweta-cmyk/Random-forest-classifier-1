from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final = [np.array(features)]
    prediction = model.predict(final)[0]
    result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected"
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)