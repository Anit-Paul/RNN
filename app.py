import streamlit as st
import tensorflow as tf
import numpy as np

st.title("✅ Streamlit + TensorFlow Test App")

# Display TensorFlow version
st.write("TensorFlow version:", tf.__version__)

# Create a simple dummy model (Dense layer)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])
model.compile(optimizer='adam', loss='mse')

st.subheader("🔢 Enter a number to make a prediction")

# User input
x = st.number_input("Input:", value=1.0)

# Make prediction
if st.button("Predict"):
    pred = model.predict(np.array([[x]]))
    st.success(f"Prediction: {pred[0][0]:.4f}")
