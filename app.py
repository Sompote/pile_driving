import numpy as np
import torch
from joblib import load
import gradio as gr

# Load the StandardScaler from the file
scaler = load('scaler1.pkl')
scaler2 = load('scaler2.pkl')

# Load a TorchScript model
model = torch.jit.load('model.pt')
model.eval()

def predict_particle_velocity(pile_width, pile_length, weight, drop_height, distance, location, trigger):
    # Prepare input
    X_t = np.array([pile_width, pile_length, weight, drop_height, distance, location + 1, trigger + 1])
    X_t = np.reshape(X_t, (1, -1))
    
    # Scale input
    X_t = scaler.transform(X_t)
    X_t = torch.tensor(X_t, dtype=torch.float32)
    
    # Make prediction
    with torch.no_grad():
        test_outputs = model(X_t)
        # Inverse transform using the scaler
        test_outputs2 = scaler2.inverse_transform(test_outputs)
    
    return f"Particle velocity = {test_outputs2[0,0]:.2f} mm/s"

# Define Gradio interface
iface = gr.Interface(
    fn=predict_particle_velocity,
    inputs=[
        gr.Number(label="Pile width (mm)", value=300),
        gr.Number(label="Pile length (m)", value=18),
        gr.Number(label="Weight (ton)", value=4.2),
        gr.Number(label="Drop height (m)", value=0.5),
        gr.Number(label="Distance (m)", value=30),
        gr.Radio(["On ground", "On foundation", "On building"], label="Location", type="index", value="On ground"),
        gr.Radio(["Longitudinal", "Transverse", "Vertical"], label="Trigger", type="index", value="Longitudinal")
    ],
    outputs="text",
    title="Peak Particle Velocity Estimator for Bangkok Sub-soil",
    description="Estimate the peak particle velocity based on input parameters."
)

# Launch the interface
iface.launch()