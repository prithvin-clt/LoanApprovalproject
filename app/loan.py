import streamlit as st
import pickle
import numpy as np
from PIL import Image
from scipy.sparse import csr_matrix

def load_model():
    with open("model.sav", "rb") as file:
        model = pickle.load(file)
    return model

def load_encoders():
    with open("LabelEncoder.sav", "rb") as file:
        label_encoder = pickle.load(file)
    with open("ohe2.sav", "rb") as file:
        ohe2 = pickle.load(file)
    with open("ohe3.sav", "rb") as file:
        ohe3 = pickle.load(file)
    with open("Scalar.sav", "rb") as file:
        scalar = pickle.load(file)
    return label_encoder, ohe2, ohe3, scalar

def preprocess_input(data, label_encoder, ohe2, ohe3, scalar):
    try:
        # Extract numerical values
        applicant_income = float(data[0])
        loan_amount = float(data[1])
        loan_term = float(data[2])

        # Convert categorical data
        credit_history_encoded = np.array([[1 if data[3] == "Yes" else 0]])
        
        # Handle unknown categories safely
        gender_array = np.array([[data[4]]])
        if data[4] not in ohe2.categories_[0]:
            gender_array = np.array([[ohe2.categories_[0][0]]])
        
        married_array = np.array([[data[5]]])
        if data[5] not in ohe3.categories_[0]:
            married_array = np.array([[ohe3.categories_[0][0]]])

        gender_encoded = ohe2.transform(gender_array).toarray()
        married_encoded = ohe3.transform(married_array).toarray()

        # Ensure correct feature alignment and padding
        expected_gender_features = ohe2.transform(ohe2.categories_[0].reshape(-1, 1)).shape[1]
        expected_married_features = ohe3.transform(ohe3.categories_[0].reshape(-1, 1)).shape[1]

        if gender_encoded.shape[1] < expected_gender_features:
            gender_encoded = np.pad(gender_encoded, ((0, 0), (0, expected_gender_features - gender_encoded.shape[1])), mode='constant')

        if married_encoded.shape[1] < expected_married_features:
            married_encoded = np.pad(married_encoded, ((0, 0), (0, expected_married_features - married_encoded.shape[1])), mode='constant')

        # Combine numerical and encoded categorical features
        processed_data = np.hstack((np.array([[applicant_income, loan_amount, loan_term]]), 
                                    credit_history_encoded, gender_encoded, married_encoded))

        # Ensure expected feature size by adding missing zero columns
        expected_features = 22  # Expected number of features
        if processed_data.shape[1] < expected_features:
            processed_data = np.pad(processed_data, ((0, 0), (0, expected_features - processed_data.shape[1])), mode='constant')

        # Apply scaling
        processed_data = scalar.transform(processed_data)  # Ensure proper reshaping

        st.write("Processed Data Shape:", processed_data.shape)
        st.write("Processed Data Values:", processed_data)
        
        return processed_data
    except Exception as e:
        st.error(f"Error in data preprocessing: {e}")
        return None

st.markdown(
        """
        <style>
        body, .stApp {
            background: linear-gradient(to right, #ffffff, #ccffcc);
            color: black !important;
        }
        h1 {
            color: #008000 !important;
            text-align: center;
            font-size: 3rem;
            font-weight: bold;
            text-transform: uppercase;
        }
        h2, h3, h4, p, label, .stTextInput, .stSelectbox, .stNumberInput, .stButton>button {
            color: black !important;
        }
        .stButton>button {
            background-color: #008000;
            color: white;
            border-radius: 10px;
            font-size: 1.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True
)
    

def main():
    st.title("Loan Approval Prediction")
    image = Image.open("Loan_Term.png")
    st.image(image, width=800)
    
    model = load_model()
    label_encoder, ohe2, ohe3, scalar = load_encoders()
    
    st.header("Enter Loan Application Details")
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income", min_value=0)
    credit_history = st.selectbox("Credit History", ["No", "Yes"])
    loan_amount = st.number_input("Loan Amount", min_value=0)
    loan_term = st.number_input("Loan Term (in months)", min_value=0)
    
    if st.button("Predict Loan Approval"):
        input_data = [applicant_income, loan_amount, loan_term, credit_history, gender, married]
        processed_data = preprocess_input(input_data, label_encoder, ohe2, ohe3, scalar)
        
        if processed_data is not None:
            prediction = model.predict(processed_data)
            
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(processed_data)
                st.write("Prediction Probability:", proba)
            
            st.write("Prediction Value:", prediction)
            
            if prediction[0] == 1:
                st.success("🎉Congratulations! Your loan is approved.🎉")
            else:
                st.error("❌Sorry, your loan application is not approved.❌")

if __name__ == "__main__":
    main()
