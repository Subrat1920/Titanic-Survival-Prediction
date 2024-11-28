import streamlit as st
import pickle
import numpy as np

# Load the StandardScaler and Logistic Regression Pickle Models
model = pickle.load(open('Models/model.pkl', 'rb'))  # Logistic Regression model
scaler = pickle.load(open('Models/scaler.pkl', 'rb'))  # StandardScaler

# Initializing Streamlit App
def main():
    st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")
    st.title('🚢 Titanic Survival Prediction')
    st.write(
        """
        Predict whether a passenger survived or not based on:
        - Passenger Class (Pclass)
        - Gender (Sex)
        - Age
        - Number of Siblings/Spouse (Siblings)
        - Number of Parents/Children (Parch)
        - Fare Price (Fare)
        - Port of Embarkation (Embarked)
        - Multiple Cabin indicator
        """
    )
    
    st.header("Input Passenger Details")
    
    # Input Form
    pclass = st.selectbox('Passenger Class (Pclass)', [1, 2, 3], index=0)
    sex = st.selectbox('Gender', ['Female (0)', 'Male (1)'], index=0)
    age = st.number_input('Age', min_value=0, max_value=100, value=25)
    siblings = st.slider('Number of Siblings/Spouses (Siblings)', 0, 5, 0)
    parch = st.slider('Number of Parents/Children (Parch)', 0, 6, 0)
    fare = st.number_input('Fare Amount', min_value=0.0, max_value=1000.0, value=10.0)
    embarked = st.selectbox('Port of Embarkation', ['C = Cherbourg (0)', 'Q = Queenstown (1)', 'S = Southampton (2)'], index=2)
    multiple_cabin = st.slider('Multiple Cabin Indicator', 0, 4, 0)
    
    # Mapping input features
    sex = 0 if 'Female' in sex else 1
    embarked = ['C', 'Q', 'S'].index(embarked.split('=')[0].strip())

    if st.button('Predict'):
        # Feature list
        features = [pclass, sex, age, siblings, parch, fare, embarked, multiple_cabin]
        
        try:
            # Scaling the features
            scaled_features = scaler.transform([features])
            
            # Model Prediction
            result = model.predict(scaled_features)
            
            # Display the result
            if result[0] == 1:
                st.success('🟢 Passenger Survived')
            else:
                st.error('🔴 Passenger Died')
        except Exception as e:
            st.error(f"Error occurred: {e}")

# Run the app
if __name__ == '__main__':
    main()
