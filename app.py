import streamlit as st
import joblib


pipe_lr = joblib.load(open("rf(8000).pkl","rb"))

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_pred_prob(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def main():
    st.title("Sentiment Analysis")
    st.subheader("Home-Emotion")

    with st.form(key='emotion clf form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        col1,col2 = st.columns(2)

        # Apply functions here 
        prediction = predict_emotions(raw_text)
        probability = get_pred_prob(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            st.write(prediction)

        with col2:
            st.success("Prediction probability")
            st.write(probability)

    




if __name__ == '__main__':
    main()