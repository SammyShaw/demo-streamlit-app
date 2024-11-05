#### DSI Streamlit Web App Tutorial ####

"""
For a full list of steamlit commands use streamlit.com -> documents -> etc.
"""


#### refer to Pipelines tutorial ####

# import libraries

import streamlit as st
import pandas as pd
import joblib

# load our model pipeline object
model = joblib.load("model.joblib")


# Add title and instructions

st.title("Purchase Prediction Model")
st.subheader("Enter Customer Information and submit for likelihood to purchase")

#### lets look at what we have so far:
    # Open Anaconda Prompt
    # Make sure we're in the right evnironment - dsi streamlit (in anaconda prompt window):
        # >conda activate dsi-streamlit-web-app
    # point to the directory where our code is located:
        # cd C:\Dat_Sci\DSI files\Model Deployment\Streamlit
    # use a special streamlit command to run the app locally: 
        # streamlit run dsi-streamlit-web-app.py


############ Input forms ################
# assign the logic to an object

# age input form
age = st.number_input(
    label = "01. Enter the customer's age",
    min_value = 18,
    max_value = 120, 
    value = 35
    )

# gender input form
gender = st.radio(
    label = "02. Enter the customer's gender",
    options = ["M","F"]
    )


# credit score input form
credit_score = st.number_input(
    label = "03. Enter the customer's credit score",
    min_value = 0,
    max_value = 1000, 
    value = 500
    )


# submit inputs to model 

# st.button("Submit for Prediction")

if st.button("Submit for Prediction"):
    
    # store our data in a dataframe for prediction
    new_data = pd.DataFrame({"age" : [age], "gender" : [gender], "credit_score" : [credit_score]})
    
    # apply model pipeline to new data and extract probability prediction
    pred_proba = model.predict_proba(new_data)[0][1] # because the actual probabilities are nested 
    
    # output prediction
    st.subheader(f"Based on these customer attributes, our model predicts a purchase probability of {pred_proba:.0%}")






















