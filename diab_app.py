import numpy as np
import pickle
import streamlit as st
import warnings
warnings.filterwarnings('ignore')


#loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

#create a function for prediction
def diabetes_prediction(input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
     return('The person is not diabetic')
    else:
     return('The person is diabetic')
    
def main():
  #giving a title     
  st.title('Diabetes prediction Web app')

  #getting the input data from the user
  Pregnancies = st.slider('Number of Pregnancies',0,17)
  Glucose = st.slider('Glucose level',0,200)
  BloodPressure = st.slider('Blood Pressure value',0,122)
  SkinThickness = st.slider('Skin Thickness value', 0, 99)
  Insulin = st.slider('Insulin Level', 0, 846)
  BMI = st.slider('BMI value', 0, 67)
  DiabetesPedigreeFunction = st.slider('Diabetes Pedigree value', 0.05, 2.5)
  Age = st.slider('Age of the person', 21, 81)
  
  #code for prediction
  diagnosis = ''

  #creating a button for prediction
  if st.button('Diabetes_Test_result'):
    diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])

  st.success(diagnosis)


if __name__  == '__main__':
  main()