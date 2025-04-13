import streamlit as st
import pickle

# Load trained model & vectorizer
model_file = "spam_classifier.pkl"

if not model_file:
    st.error("❌ Model file not found! Train and save the model first.")
else:
    with open(model_file, "rb") as f:
        cv, model = pickle.load(f)

    def predict(message):
        input_message = cv.transform([message]).toarray()
        prediction = model.predict(input_message)
        return "Spam" if prediction[0] == 1 else "Not Spam"

    # Streamlit UI
    st.title("📩 Spam Classifier")
    user_input = st.text_area("Enter a message:", "")

    if st.button("Check"):
        result = predict(user_input)
        st.success(f"Prediction: **{result}**")

