import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import gradio as gr
import pandas as pd

# 1. Generate Synthetic "Motor Vibration" Data
def generate_data(seq_length=50):
    t = np.linspace(0, 10, seq_length)
    # Normal sine wave vibration
    normal = np.sin(t) + np.random.normal(0, 0.1, seq_length)
    # Anomalous vibration (simulating bearing wear)
    anomalous = np.sin(t) + np.random.normal(0, 0.5, seq_length) + 0.5 * np.sin(5*t)
    return normal, anomalous

# 2. Build an LSTM Autoencoder
# This learns what "normal" looks like and flags deviations.
model = models.Sequential([
    layers.Input(shape=(50, 1)),
    layers.LSTM(32, activation='relu', return_sequences=False),
    layers.RepeatVector(50),
    layers.LSTM(32, activation='relu', return_sequences=True),
    layers.TimeDistributed(layers.Dense(1))
])

model.compile(optimizer='adam', loss='mae')

# Quick "training" on normal patterns
print("Calibrating baseline motor harmonics...")
X_normal = np.array([generate_data()[0] for _ in range(1000)]).reshape(-1, 50, 1)
model.fit(X_normal, X_normal, epochs=5, verbose=0)

# 3. Interactive Diagnostic Function
def monitor_vibration(mode):
    normal_wave, failing_wave = generate_data()
    data = failing_wave if mode == "Simulate Bearing Wear" else normal_wave
    
    # Predict/Reconstruct
    reconstruction = model.predict(data.reshape(1, 50, 1)).flatten()
    loss = np.mean(np.abs(data - reconstruction))
    
    # Health Score Logic
    status = "HEALTHY" if loss < 0.25 else "CRITICAL: ANOMALY DETECTED"
    
    # Prepare Plotting Data
    df = pd.DataFrame({"Time": np.arange(50), "Vibration": data})
    return df, f"Loss Score: {loss:.4f}", status

# 4. Gradio Dashboard
with gr.Blocks() as demo:
    gr.Markdown("## 🛠️ Deep Learning Predictive Maintenance Dashboard")
    with gr.Row():
        input_mode = gr.Radio(["Normal Operation", "Simulate Bearing Wear"], label="Motor State")
        btn = gr.Button("Analyze Live Feed")
    
    with gr.Row():
        plot = gr.LinePlot(x="Time", y="Vibration", title="Sensor Waveform")
        with gr.Column():
            label = gr.Textbox(label="Reconstruction Loss")
            output = gr.Label(label="System Status")

    btn.click(monitor_vibration, inputs=input_mode, outputs=[plot, label, output])

if __name__ == "__main__":
    demo.launch()
