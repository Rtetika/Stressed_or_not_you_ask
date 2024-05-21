import joblib
import streamlit as st

file_name = "finalized_model.sav"
loaded_model = joblib.load(file_name)

st.title("Stress Analysis")
# st.header("Stress Analysis")

st.markdown("Share with us what are you feeling these days. We are here for you")

# removing the streamlit banner at the bottom
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.text_input("Text", key="user_text")

# You can access the value at any point with:
text = st.session_state.user_text
print(text)

result = loaded_model.predict([text])

st.write(result[0])

