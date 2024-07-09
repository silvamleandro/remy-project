# Imports
from keras import Sequential, utils
from keras.layers import Dense


# Set random state
RANDOM_STATE = 42

# Set random seed in Keras
utils.set_random_seed(RANDOM_STATE)



def create_model(input_dim):
    model = Sequential([ # Sequential autoencoder
        Dense(units=32, activation='relu', input_dim=input_dim), # Encoder
        Dense(units=16, activation='relu'),
        Dense(units=8, activation='relu'),
        Dense(units=4, activation='relu'),
        Dense(units=8, activation='relu'), # Decoder
        Dense(units=16, activation='relu'),
        Dense(units=32, activation='relu'),
        Dense(units=input_dim, activation='sigmoid')
    ])
    # Autoencoder compile
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model # Return autoencoder model
