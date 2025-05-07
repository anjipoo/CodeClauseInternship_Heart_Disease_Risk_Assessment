import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(page_title="heart disease prediction",layout="wide")

numcols=['age','trestbps','chol','thalch','oldpeak']
model=joblib.load('models/best_model.pkl')
scaler=joblib.load('models/scaler.pkl')
with open('models/metrics.json','r') as f:
    metrics=json.load(f)

st.title("heart disease prediction app")
st.write("this app detects the likelihood of having a heart disease based on parameters")

st.header("model performance")
st.write("accuracy:",metrics['accuracy'])
st.write("precision",metrics['precision'])
st.write("recall",metrics['recall'])
st.write("f1-score:",metrics['f1'])

st.header("patient input data")
col1,col2=st.columns(2)

with col1:
    age=st.number_input("Age",min_value=0,max_value=120,value=40)
    sex=st.selectbox("Sex",options=[0,1],format_func=lambda x:"Female" if x==0 else "Male")
    cp=st.selectbox("Chest Pain Type",options=[0,1,2,3],format_func=lambda x:["Typical Angina","Atypical Angina","Non-Anginal Pain","Asymptomatic"][x])
    trestbps=st.number_input("Resting Blood Pressure (mm Hg)",min_value=0,max_value=300,value=120)
    chol=st.number_input("Serum Cholesterol (mg/dL)",min_value=0,max_value=600,value=200)
    fbs=st.selectbox("Fasting Blood Sugar > 120 mg/dL",options=[0,1],format_func=lambda x:"No" if x==0 else "Yes")

with col2:
    restecg=st.selectbox("Resting ECG Results",options=[0,1,2],format_func=lambda x:["Normal","ST-T Wave Abnormality","Left Ventricular Hypertrophy"][x])
    thalch=st.number_input("Maximum Heart Rate Achieved",min_value=0,max_value=200,value=150)
    exang=st.selectbox("Exercise Induced Angina",options=[0,1],format_func=lambda x:"No" if x==0 else "Yes")
    oldpeak=st.number_input("ST Depression (oldpeak)",min_value=0.0,max_value=10.0,value=1.0,step=0.1)
    slope=st.selectbox("Slope of Peak Exercise ST Segment",options=[0,1,2],format_func=lambda x:["Upsloping","Flat","Downsloping"][x])

if st.button("predict"):
    input_data=pd.DataFrame({
        'age':[age],
        'sex':[sex],
        'cp':[cp],
        'trestbps':[trestbps],
        'chol':[chol],
        'fbs':[fbs],
        'restecg':[restecg],
        'thalch':[thalch],
        'exang':[exang],
        'oldpeak':[oldpeak],
        'slope':[slope],
    })

    input_data[numcols]=scaler.transform(input_data[numcols])

    prediction=model.predict(input_data)[0]
    probability=model.predict_proba(input_data)[0]

    st.header("prediction result")
    result="heart disease" if prediction==1 else "no heart disease"
    st.write(result)
    st.write(probability[prediction])
    if probability[prediction]<0.7:
        st.warning("low confidence - consider consulting a doctor")

st.header("data visualizations")

st.subheader("target distribution")
st.image('visualizations/target_dist.png')
st.subheader("numerical feature distribution")
st.image('visualizations/num_histogram.png')
st.subheader("feature differences by heart disease")
st.image('visualizations/box_plots.png')
st.subheader("correlation heatmap")
st.image('visualizations/corr_heatmap.png')
st.subheader("feature importance")
st.image('visualizations/ft_importance.png')