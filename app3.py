import streamlit as st
import torch
import joblib
import numpy as np

# Load the model and scalers
@st.cache_resource
def load_model_and_scalers():
    try:
        loaded_model = torch.jit.load('model.pt')
        loaded_scaler1 = joblib.load('scaler1.pkl')
        loaded_scaler2 = joblib.load('scaler2.pkl')
        return loaded_model, loaded_scaler1, loaded_scaler2
    except Exception as e:
        st.error(f"Error loading model or scalers: {str(e)}")
        return None, None, None

loaded_model, loaded_scaler1, loaded_scaler2 = load_model_and_scalers()

st.title('Peak Particle Velocity Prediction App')

# Create input fields
st.header('Enter Input Values:')
pile_width = st.number_input('Pile width (mm)', value=300.0)
pile_length = st.number_input('Pile length (m)', value=18.0)
weight = st.number_input('Weight (ton)', value=4.2)
drop_height = st.number_input('Drop height (m)', value=0.5)
distance = st.number_input('Distance (m)', value=30.0)
location = st.selectbox('Location', ['On ground', 'On foundation', 'On building'], index=0)
trigger = st.selectbox('Trigger', ['Longitudinal', 'Transverse', 'Vertical'], index=0)

# Convert location and trigger to numerical values
location_value = ['On ground', 'On foundation', 'On building'].index(location) + 1
trigger_value = ['Longitudinal', 'Transverse', 'Vertical'].index(trigger) + 1

# Button to make prediction
if st.button('Make Prediction'):
    if loaded_model is None or loaded_scaler1 is None or loaded_scaler2 is None:
        st.error("Model or scalers failed to load. Please check your files and try again.")
    else:
        try:
            # Prepare input data
            input = np.array([pile_width, pile_length, weight, drop_height, distance, location_value, trigger_value])
            inputx = np.reshape(input, (1, 7))
            
            # Transform input data
            X_test1 = loaded_scaler1.transform(inputx).astype(np.float32)
            X_test1 = torch.from_numpy(X_test1)
            
            # Make prediction
            with torch.no_grad():
                test_outputs = loaded_model(X_test1)
                test_outputs2 = loaded_scaler2.inverse_transform(test_outputs.cpu())
            
            # Display results
            st.subheader('Prediction Results:')
            st.write(f"Peak Particle Velocity: {test_outputs2[0][0]:.2f} mm/s")
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

st.sidebar.header('About')
st.sidebar.info('This app uses a pre-trained PyTorch model to predict peak particle velocity based on user input. It is specifically designed for Bangkok sub-soil conditions.')