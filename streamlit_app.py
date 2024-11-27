## import neccesary packages and library
import streamlit as st
import pickle 
import numpy as np


## load the StandardScaler and Logistic Regression Pickle Models
model = pickle.load(open('Models/model.pkl', 'rb')) # for logistic regression
scaler = pickle.load(open('Models/scaler.pkl', 'rb')) # for standard scaler

## initializing streamlit app
def main():
    st.title('Titanic Survival Prediction')
    st.write('This app predicts wheter the passenger survived based on the data like Age, Pclass, Multiple Cabin, Sex, Siblings, Parental-Children, Fare Rate and Embarked')
    st.write('Please select the values for the features')


    ## form for input features
    
    pclass = st.number_input('Pclass')
    sex = st.number_input('Sex (female: 0 and male: 1)')
    age = st.number_input('Age')
    siblings = st.number_input('Siblings [0-5] ')
    parch = st.number_input('Parch [0-6]')
    fare = st.number_input('Fare')
    embarked = st.number_input('Emabrked [ C : 0, Q : 1, S : 2]')
    multiple_cabin = st.number_input('Multiple Cabin [0-4]')

    if st.button('Predict'):
        ## creating a list with all input features
        features = [pclass, sex, age, siblings, parch, fare, embarked, multiple_cabin]

        scaled_features = scaler.transform([features])
        
        result = model.predict(scaled_features)

        if result[0]==1:
            st.subheader('Passenger Survived')
        elif result[0] == 0:
            st.subheader('Passenger Died')
        else:
            st.subheader('Error in input values')

if __name__=='__main__':
    main()
