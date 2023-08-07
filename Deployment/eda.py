import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

def run():
    # Membuat title
    st.title('Churn Prediction')

    # Membuat Sub Header
    st.subheader('Exploratory Data Analysis')

    # Menambahkan Deskripsi
    st.write('App ini dibuat untuk memprediksi apakah seseorang akan tetap menjadi pelanggan atau tidak.')
    # Membuat Garis Lurus
    st.markdown('---')
    st.write('Dataset yang digunakan adalah Churn Dataset, dataset ini berisi 37010 baris dan 22 kolom')

    # Show Dataset
    data = pd.read_csv('churn.csv')
    st.dataframe(data)
    st.write('Created by: [Fitri Octaviani]')

if __name__ == '__main__':
    run()