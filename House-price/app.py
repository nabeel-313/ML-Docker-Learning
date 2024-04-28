from flask import Flask, jsonify, request
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def do_prediction():
    json = request.get_json()
    #print(json)
    scaler = joblib.load('Model/std_scaler.joblib')
    model = joblib.load('Model/rf_reg.joblib')
    df = pd.DataFrame(json, index=[0])
    #print(df)

    
  

    df_x_scaled = scaler.transform(df)

    df_x_scaled = pd.DataFrame(df_x_scaled, columns=df.columns)
    y_predict = model.predict(df_x_scaled)

    result = {"Predicted House Price" : y_predict[0]}
    return jsonify(result)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8000, debug=True)