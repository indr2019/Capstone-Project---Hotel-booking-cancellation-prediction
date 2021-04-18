import pandas as pd
from flask import Flask, jsonify, request
import joblib 

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    req = request.get_json()
    input_data = req['data']
    input_data_df = pd.DataFrame.from_dict(input_data)

    model = joblib.load('model.pkl')
    #scale_obj = joblib.load('scale.pkl')
    #input_data_scaled = scale_obj.transform(input_data_df)
    #print(input_data_scaled)

    prediction = model.predict(input_data_df)

    if prediction[0] == 1:
        hotel_booking = 'Cancelled'
    else:
        hotel_booking = 'Not cancelled'

    return jsonify({'output':{'hotel_booking':hotel_booking}})


@app.route('/')
def home():
    return "Welcome to Hotel booking reservation system"

    
     
if __name__=='__main__':
    app.run(host='0.0.0.0', port='3000')
        

