#!/usr/bin/env python
# coding: utf-8

# In[77]:



from flask import Flask,render_template,request
import numpy as np
import joblib
import pandas as pd

app=Flask(__name__)


# Load models
model_reg = joblib.load("weader_model_reg.pkl")
model_lasso = joblib.load("weader_model_lasso.pkl")
model_ridge = joblib.load("weader_model_ridge.pkl")

# Load scaler
scaler = joblib.load("weather_scaler.pkl")

location_mapping = {
    "New York": 3,
    "London": 2,
    "Tokyo": 7,
    "Paris": 4,
    "Sydney": 6,
    "Dubai": 0,
    "Rome": 5,
    "Hong Kong": 1
}
def get_location_index(location_name):
    return location_mapping[location_name]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])   
def predict():
    humidity = float(request.form["humidity"])
    visibility = float(request.form["visibility"])
    feels_like = float(request.form["feels_like"])
    cloud_cover = float(request.form["cloud_cover"])
    location = request.form["location"]
    
    location_index=get_location_index(location)
    

    input_d = [[feels_like,humidity, cloud_cover, visibility, location_index]]
    column= ['Feels Like (°C)', 'Humidity (%)', 'Cloud Cover (%)', 'Visibility (km)', 'Location_index']
    
      # Scale inputs
    
    # Convert to dataframe
    input_data=pd.DataFrame(input_d ,columns=column)
    scaled_data = scaler.transform(input_data)


    # Predict
    prediction = model_reg.predict(scaled_data)[0]

    return render_template("index.html", prediction=round(prediction, 2))


if __name__=="__main__":
    app.run(debug=True)


# In[ ]:





# In[ ]:





# In[ ]:




