import streamlit as st
import eda
import prediction

navigation = st.sidebar.selectbox('Page Navigation: ',('EDA','Customer Churn Prediction'))

if navigation == 'EDA':
    eda.run()
else:
    prediction.run()