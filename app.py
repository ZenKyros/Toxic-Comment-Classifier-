import streamlit as st
import joblib


model = joblib.load('toxic_nb_model.pkl')

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def stemming(text):

    return text


st.title("Toxic Comment Classifier")


text = st.text_area("Enter a comment:")


if st.button("Predict"):
    processed = stemming(text)
    prediction = model.predict([processed])
    
    st.write("Raw prediction:", prediction)
    st.write("Type:", type(prediction))
 
    try:
        pred_list = prediction[0]
    except Exception as e:
        st.error(f"Error: {e}")
        pred_list = []

    for label, result in zip(labels, pred_list):
        st.write(f"**{label}**: {'Yes' if result == 1 else 'No'}")
