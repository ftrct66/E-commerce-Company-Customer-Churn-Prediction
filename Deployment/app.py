import streamlit as st
import pandas as pd
import numpy  as np
import joblib


st.header('Model Deployment Milestone 2 Phase 1')
st.write("""
Created by Fitri Octaviani
Hck 006.
""")


# load best model
rf_tuning = joblib.load('rf_model1.pkl')


# Construct Data Infer
# define semua fitur/kolom
features = ['Total_Protiens','Albumin','Albumin_and_Globulin_Ratio']

def infer(data_infer):
    # predict result random forest model
    y_pred_rf = rf_tuning.predict(data_infer)
    return y_pred_rf 

st.header("Prediksi pasien apakah memiliki penyakit liver atau tidak")

# artificial data infer
Total_Protiens                  = st.number_input("Total protein dalam darah pasien :", 0.0, 100.0)                
Albumin                         = st.number_input("Kadar albumin dalam darah pasien:",0.0, 100.0)
Albumin_and_Globulin_Ratio      = st.number_input("Rasio albumin dan globulin dalam darah pasien:",0.0, 100.0)

if st.button("Predict"):
    D = {
    'Total_Protiens':Total_Protiens,
    'Albumin':Albumin,                    
    'Albumin_and_Globulin_Ratio':Albumin_and_Globulin_Ratio,            
    }
    
    # construct data inference dalam dataframe
    data_infer = pd.DataFrame(data=D,columns=features,index=[0])

    # panggil fungsi inference
    rf_pred = infer(data_infer)
    res_rf = ''

    # interpretasi hasil prediksi
    if rf_pred[0] == 0:
        res_rf = "Kemungkinan memiliki penyakit liver"
    else:
        res_rf = "Kemungkinan tidak memiliki penyakit liver"

    st.header(f"Hasil Prediksi: ")
    st.write(res_rf)