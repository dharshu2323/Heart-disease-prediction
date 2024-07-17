import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']
        print(data)
        # Convert the data into the right format
        input_data = np.array(list(data.values())).reshape(1, -1)
        print(input_data)
        
        # Transform the data using the scaler
        new_data = scalar.transform(input_data)
        print(new_data)
        
        # Predict using the loaded model
        output = regmodel.predict(new_data)
        print(output[0])
        
        # Convert the output to a standard Python integer
        result = int(output[0])
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
