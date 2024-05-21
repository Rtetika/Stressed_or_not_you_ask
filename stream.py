import joblib
import streamlit as st
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer



# Load the trained model and the CountVectorizer
file_name = "finalized_model.sav"
loaded_model = joblib.load(file_name)
cv = joblib.load('count_vectorizer.pkl')  # Assuming you saved the CountVectorizer as well

# Streamlit app title
st.title("Stress Analysis")

# Markdown instruction
st.markdown("Share with us what are you feeling these days. We are here for you")

# Removing the Streamlit banner at the bottom
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Text input for user
user_text = st.text_input("Text", key="user_text")

# Access the value and handle prediction
if user_text:
    try:
        # Preprocess the input text
        cleaned_text = clean(user_text)
        
        # Vectorize the input text
        vectorized_text = cv.transform([cleaned_text])
        
        # Predict the result
        result = loaded_model.predict(vectorized_text)
        
        # Display the result
        st.write(f"Prediction: {result[0]}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
