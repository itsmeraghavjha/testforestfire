import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler



application=Flask(__name__)
app=application


#import ridge regressor and standard scaler pickle

ridge_model=pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler=pickle.load(open('models/scaler.pkl', 'rb'))



@app.route("/")
def index():
    return render_template('index.html')


@app.route("/predictdata", methods=["GET","POST"])
def predict_datapoint():
    if request.method=="POST":
        temperature = float(request.form.get("Temperature"))
        rh = float(request.form.get("RH"))
        ws = float(request.form.get("Ws"))
        rain = float(request.form.get("Rain"))
        ffmc = float(request.form.get("FFMC"))
        dmc = float(request.form.get("DMC"))
        isi = float(request.form.get("ISI"))
        classes = float(request.form.get("Classes"))
        region = float(request.form.get("Region"))
        print(temperature, rh, ws, rain, ffmc, dmc, isi, classes, region)
        input_df = pd.DataFrame([{
        "Temperature": temperature,
        "RH": rh,
        "Ws": ws,
        "Rain": rain,
        "FFMC": ffmc,
        "DMC": dmc,
        "ISI": isi,
        "Classes": classes,
        "Region": region
    }])
        
        print(input_df)

        new_data_scaled=standard_scaler.transform(input_df)
        Prediction = ridge_model.predict(new_data_scaled)
        print("Prediction-->",Prediction)
        return render_template("home.html", prediction=Prediction[0])
    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run("0.0.0.0")