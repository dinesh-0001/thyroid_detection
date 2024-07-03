from flask import Flask, request, render_template
import pickle
import os
import joblib

app = Flask(__name__)

kmeans = pickle.load(open("../Cluster_model/clustering.pkl", "rb"))

encoder = joblib.load(open("../Encoder/label_encoder_class.joblib", "rb"))

def find_model(cluster):
    if(cluster == 0):
        with open('../Models/CLF0/clf.pkl', 'rb') as f:
            model = pickle.load(f)
    elif(cluster == 1):
        with open('../Models/KNN1/knn.pkl', 'rb') as f:
            model = pickle.load(f)
    else:
        with open('../Models/DT2/dt.pkl', 'rb') as f:
            model = pickle.load(f)
    return model

@app.route("/", methods=['GET'])
def index():
    return render_template("main.html")

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form["age"])
        T3 = float(request.form["T3"])
        TT4 = float(request.form["TT4"])
        T4U = float(request.form["T4U"])
        FTI = float(request.form["FTI"])
        sex_ = request.form['sex']
        if (sex_ == "Male"):
            sex = 1
        else:
            sex = 0

        thyroxine = request.form["on_thyroxine"]
        if (thyroxine == "true"):
            on_thyroxine = 1
        else:
            on_thyroxine = 0

        query_thyroxine = request.form["query_thyroxine"]
        if (query_thyroxine == "true"):
            query_on_thyroxine = 1
        else:
            query_on_thyroxine = 0
        
        antithyroid_medication = request.form["antithyroid_medication"]
        if( antithyroid_medication == "true"):
            on_antithyroid_medication = 1
        else:
            on_antithyroid_medication = 0

        I131_treatment_ = request.form["I131_treatment"]
        if(I131_treatment_ == "true"):
            I131_treatment = 1
        else:
            I131_treatment = 0

        query_hypothyroid_ = request.form["query_hypothyroid"]
        if(query_hypothyroid_ == "true"):
            query_hypothyroid = 1
        else:
            query_hypothyroid = 0
        
        hypopituitary_ = request.form["hypopituitary"]
        if(hypopituitary_ == "true"):
            hypopituitary = 1
        else:
            hypopituitary = 0

        psych_ = request.form["psych"]
        if(psych_ == "true"):
            psych = 1
        else:
            psych = 0

        sick_ = request.form['sick']
        if (sick_ == 'true'):
            sick = 1
        else:
            sick = 0

        lithium_ = request.form["lithium"]
        if(lithium_ == "true"):
            lithium = 1
        else:
            lithium = 0

        pregnant_ = request.form['pregnant']
        if (pregnant_ == 'true'):
            pregnant = 1
        else:
            pregnant = 0

        thyroid_surgery_ = request.form['thyroid_surgery']
        if (thyroid_surgery_ == 'true'):
            thyroid_surgery = 1
        else:
            thyroid_surgery = 0

        goitre_ = request.form['goitre']
        if(goitre_ == 'true'):
            goitre = 1
        else:
            goitre = 0

        tumor_ = request.form['tumor']
        if (tumor_ == 'true'):
            tumor = 1
        else:
            tumor = 0

        cluster_output = kmeans.predict([[age,
                                    sex, on_thyroxine, query_on_thyroxine, on_antithyroid_medication, sick, pregnant, thyroid_surgery,  I131_treatment,  
                                     query_hypothyroid, query_hypothyroid, lithium,  goitre, tumor, hypopituitary, psych, T3,
                                     TT4,
                                     T4U,
                                     FTI]])
        
        cluster_num = cluster_output

        model = find_model(cluster_num)
        
        prediction_output = model.predict([[age,
                                    sex, on_thyroxine, query_on_thyroxine, on_antithyroid_medication, sick, pregnant, thyroid_surgery,  I131_treatment,  
                                     query_hypothyroid, query_hypothyroid, lithium,  goitre, tumor, hypopituitary, psych, T3,
                                     TT4,
                                     T4U,
                                     FTI
                                    ]])
        
        prediction = int(prediction_output[0])

        encoded_prediction = encoder.inverse_transform([prediction])

        return f"<div style='display: flex; justify-content: center; align-items: center; height: 100vh;'><h1>You have {encoded_prediction[0]}</h1></div>"


if __name__ == '__main__':
    app.run(debug=True)

