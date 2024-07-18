# Imports
from tensorflow.keras import layers, utils
import numpy as np
import tensorflow as tf

# Set random seed in Keras
utils.set_random_seed(42)


def create_model(input_dim):
    # Input layer
    input = tf.keras.layers.Input(shape=(input_dim,))

    # Encoder layer
    encoder = tf.keras.Sequential([
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(4, activation='relu'),
        # Bottleneck
        layers.Dense(2, activation='relu')])(input)

    # Decoder layer
    decoder = tf.keras.Sequential([
        layers.Dense(4, activation="relu"),
        layers.Dense(8, activation="relu"),
        layers.Dense(16, activation='relu'),
        layers.Dense(32, activation='relu'),        
        layers.Dense(input_dim, activation="sigmoid")])(encoder)
    
    # Instantiate the autoencoder
    model = tf.keras.Model(inputs=input, outputs=decoder)
    # Compile...
    model.compile(optimizer='adam', loss='mae')
    return model # Autoencoder model


def reconstruction_loss(x, x_hat):
    return np.mean(abs(x - x_hat), axis=1)  # Mean Absolute Error (MAE)


def distance_calculation(losses, normal, abnormal):
    # For each sample loss, calculate the minimun distance and set a label for test purpose
    preds = np.zeros(len(losses)) # Create array for predicted values
    for i, loss in enumerate(losses):
        if abs(loss - normal) > abs(loss - abnormal):
            preds[i] = 1 # Abnormal
        else: preds[i] = 0 # Normal

    return preds # Predicted values