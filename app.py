from flask import Flask, render_template, request,url_for
import os
import numpy as np
import pandas as pd
from Thyroid_Disease.pipeline.prediction import PredictionPipeline
from Thyroid_Disease import logger

app = Flask(__name__)

@app.route("/", methods=['GET'])
@app.route("/home", methods=['GET'])
def homePage():
    return render_template("index.html")

@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 


@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            Inputval = request.form.to_dict()
            logger.info(f"inside def index and taking values {Inputval} ")
            age =float(Inputval['age'])
            sex = 1 if Inputval['sex']=='m' else 0
            on_thyroxine = 1 if Inputval['on_thyroxine'] =='t' else 0
            query_on_thyroxine = 1 if Inputval['query_on_thyroxine'] =='t' else 0
            #logger.info(f" value taken {on_thyroxine}, {query_on_thyroxine}")
            on_antithyroid_medication = 1 if Inputval['on_antithyroid_medication']=='t' else 0
            sick = 1 if Inputval['sick']=='t' else 0
            pregnant = 1 if Inputval['pregnant']=='t' else 0
            thyroid_surgery = 1 if Inputval['thyroid_surgery']=='t' else 0
            I131_treatment = 1 if Inputval['I131_treatment']=='t' else 0
            lithium = 1 if Inputval['lithium']=='t' else 0
            #logger.info(f" taken info : {lithium} {I131_treatment} {thyroid_surgery}")
            query_hypothyroid = 1 if Inputval['query_hypothyroid']=='t' else 0
            query_hyperthyroid = 1 if Inputval['query_hyperthyroid']=='t' else 0
            goiter = 1 if Inputval['goiter']=='t' else 0
            tumor = 1 if Inputval['tumor']=='t' else 0
            hypopituitary = 1 if Inputval['hypopituitary']=='t' else 0
            psych = 1 if request.form['psych']=='t' else 0
            T3 =float(Inputval['T3'])
            TT4 =float(Inputval['TT4'])
            T4U =float(Inputval['T4U'])
            FTI =float(Inputval['FTI'])
       
         
            data = [age,sex,on_thyroxine,query_on_thyroxine,on_antithyroid_medication,sick,pregnant,
                    thyroid_surgery,I131_treatment,query_hypothyroid,query_hyperthyroid,lithium,goiter,
                    tumor,hypopituitary,psych,T3,TT4,T4U,FTI]
            logger.info(f"collected data {data}")
            data = np.array(data).reshape(1, len(data))
            logger.info(f"Collected Data: {data}")
            #print(data)
            
            result = {0:'Compensated Hypothyroid',
                      1:'Negative', 2: 'Primary Hypothyroid',
                       3: 'Secondary Hypothyroid'}
            obj = PredictionPipeline()
            predict = int(obj.predict(data))
            logger.info(f"Predicted value : '{result[predict]}', predict output: {predict}")

            return render_template('results.html', prediction = result[predict])

        except Exception as e:
            print('The Exception message is: ',e)
            return "<h1>Something went wrong</h1>"
            #return render_template("index.html")

    else:
        return render_template('index.html')







if __name__ == "__main__":
    app.run(host="0.0.0.0", port = 8080, debug=True)