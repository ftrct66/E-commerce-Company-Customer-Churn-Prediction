import streamlit as st
import pandas as pd
import numpy as np

st.title('Milestone Phase 1 Exploratory Data Analys')
st.subheader('Author : Fitri Octaviani')

# Membagi layar menjadi 2 baris dan 2 kolom
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# Memasukkan gambar ke setiap bagian
with col1:
    st.image('bar_1.png', use_column_width=True)
with col2:
    st.image('bar_2.png', use_column_width=True)
with col3:
    st.image('bar_3.png', use_column_width=True)
with col4:
    st.image('bar_4.png', use_column_width=True)

st.subheader('PHIK CORELLATION')
st.image('korelasi.png')
