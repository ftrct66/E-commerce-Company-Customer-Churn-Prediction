import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import datetime
import joblib
import json
# Load All Files

# load preprocessor
with open('preprocessor.pkl','rb') as file_1:
    preprocessor = joblib.load(file_1)

sequen_model = load_model('sequen_imp_model.h5')


def run():
  with st.form(key='Customer_Churn_Prediction'):
      st.title('Customer Churn Prediction')
      
      age = st.number_input('Age of a customer',min_value=0,max_value=99,value=46)
      gender = st.selectbox('Gender of a customer', ('Male','Female'), index=1)
      region_category = st.radio('Region that a customer belongs to',('Town', 'City','Village'))
      membership_category = st.radio('Category of the membership that a customer is using',('Premium Membership','Basic Membership','No Membership', 'Gold Membership','Silver Membership','Platinum Membership'))
      joined_through_referral= st.selectbox('Whether a customer joined using any referral code or ID', ('Yes','No'), index=1)
      preferred_offer_types = st.radio('Type of offer that a customer prefers',('Credit/Debit Card Offers','Gift Vouchers/Coupons','Without Offers'))
      medium_of_operation = st.radio('Medium of operation that a customer uses for transactions',('Smartphone','Desktop','Both'))
      internet_option = st.radio('Type of internet sevice a customer uses',('Mobile_Data','Wi-Fi','Fiber_Optic'))
      days_since_last_login = st.number_input('Number of days since a customer last logged into the website',min_value=0,max_value=31,value=16)
      avg_time_spent = st.number_input('Average time spent by a customer on the website',step=0.01,format="%.2f",min_value=0.00,max_value=9999.99,value=1447.39)
      avg_transaction_value = st.number_input('Average transaction value of a customer',step=0.01,format="%.2f",min_value=0.00,max_value=99999.99,value=11839.58)
      avg_frequency_login_days = st.number_input('Number of times a customer has logged in to the website',min_value=1, max_value=99,value=29)
      points_in_wallet = st.number_input('Points awarded to a customer on each transaction',step=0.01,format="%.2f",min_value=0.00,max_value=9999.99,value=727.91)
      used_special_discount= st.selectbox('Whether a customer uses special discounts offered', ('Yes','No'), index=1)
      offer_application_preference = st.selectbox('Whether a customer prefers offers',('No','Yes'))
      past_complaint = st.selectbox('Whether a customer has raised any complaints',('No','Yes'))
      complaint_status = st.radio('Whether the complaints raised by a customer was resolved',('Not Applicable ','Unsolved','Solved','Solved in Follow-up','No Information Available'))
      feedback = st.radio('Feedback provided by a customer',('No reason specified','Poor Product Quality','Too many ads', 'Poor Website', 'Poor Customer Service', 'Reasonable Price', 'User Friendly Website', 'Products always in Stock', 'Quality Customer Care'))
      joining_month = st.slider('Month when a customer became a member', 1, 12, 3)
      submitted = st.form_submit_button('Is the customer at risk of churning ?')

  df_inf = {
      'age': age,
      'gender': gender,
      'region_category': region_category,
      'membership_category': membership_category,
      'joined_through_referral': joined_through_referral,
      'preferred_offer_types': preferred_offer_types,
      'medium_of_operation': medium_of_operation,
      'internet_option': internet_option,
      'days_since_last_login':days_since_last_login,
      'avg_time_spent':avg_time_spent,
      'avg_transaction_value':avg_transaction_value,
      'avg_frequency_login_days':avg_frequency_login_days,
      'points_in_wallet':points_in_wallet,
      'used_special_discount':used_special_discount,
      'offer_application_preference':offer_application_preference,
      'past_complaint':past_complaint,
      'complaint_status':complaint_status,
      'feedback':feedback,
      'joining_month':joining_month

  }

  df_inf = pd.DataFrame([df_inf])
  st.dataframe(df_inf)
  
  data_inf_transform = preprocessor.transform(df_inf)

  if submitted:
      # Predict using Neural Network
      y_pred_inf = sequen_model.predict(data_inf_transform)
      
      if y_pred_inf == 0:
         st.subheader('Yes, the customer is at risk of churn')
      else:
         st.subheader('No, the customer is not at risk of churn')


st.write('Created by: Fitri Octaviani')
if __name__ == '__main__':
    run()