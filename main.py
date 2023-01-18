from flask import Flask, request, render_template
import joblib as jb
import pandas as pd
import numpy as np

app = Flask(__name__)
file = 'redwine_classifier_model.joblib'
model = jb.load(file)

@app.route('/')
def index():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():
    # Get input data from user
    input_data = np.array([float(request.form['fixed_acidity']), 
                      float(request.form['volatile_acidity']),
                      float(request.form['citric_acid']), 
                      float(request.form['residual_sugar']),
                      float(request.form['chlorides']),
                      float(request.form['free_sulfur_dioxide']),
                      float(request.form['total_sulfur_dioxide']),
                      float(request.form['density']),
                      float(request.form['pH']),
                      float(request.form['sulphates']),
                      float(request.form['alcohol'])])
    #Reshape data
    input_data = input_data.reshape(1, -1)

    
    # Make prediction
    prediction = model.predict(input_data)[0]
    

    
    # Create output message
    if prediction >= 3 and prediction <= 4:
        output_message = "This wine is bad"
        output_number = "3-4"
    elif prediction >= 5 and prediction <= 6:
        output_message = "This wine is medium"
        output_number = "5-6"
    else:
        output_message = "This wine is good"
        output_number = "7-8"

    
    return render_template("prediction.html", prediction=output_message,prediction_number=prediction,number_range = output_number)

if __name__ == "__main__":
    app.run(debug=True)

