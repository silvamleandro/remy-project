# Imports
from sklearn.model_selection import train_test_split

import numpy as np
import pickle

# Default SEED
RANDOM_STATE = 42


def simulate_spoof_location(latitude, longitude, max_offset=0.01,
                            offset_distribution="uniform", seed=RANDOM_STATE):
    # Simulate GPS spoofing by adding a random offset to latitude and longitude
    
    if seed is not None:  # Previously defined seed
        np.random.seed(seed)

    # Uniform distribution
    if offset_distribution == "uniform":
        offset_latitude = np.random.uniform(-max_offset, max_offset)
        offset_longitude = np.random.uniform(-max_offset, max_offset)
        
    # Normal distribution
    elif offset_distribution == "normal":  
        offset_latitude = np.random.normal(0, max_offset)
        offset_longitude = np.random.normal(0, max_offset)

    # Spoofed coordinate
    return latitude + offset_latitude, longitude + offset_longitude


def split_X_y(data, target_column="is_target", columns_to_drop=[]):
    # DF with the feautures
    # Target column is added here to delete in X
    # DataFrame X (features)
    X = data.drop(columns_to_drop + [target_column], axis=1)
    # y (labels)
    y = data[target_column]

    return X, y  # Matrix X and vector y


def split_train_test(df, target_column="is_target", random_state=RANDOM_STATE, test_size=0.20):
    # Split DataFrame into X and y
    X, y = split_X_y(df, target_column)
    
    # Split (100 - test_size)% of data for training/validation and (test_size)% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Add y back into X
    X_train[target_column] = y_train
    X_test[target_column] = y_test

    return X_train.reset_index(drop=True), X_test.reset_index(drop=True)  # Samples splitted into training and test


def save_object(obj_to_save, filename):
    # Save .pkl object
    pickle.dump(obj_to_save, open(f"{filename}.pkl", "wb"))


def load_object(filename):
    # Load .pkl object
    obj_to_load = pickle.load(open(f"{filename}.pkl", "rb"))

    return obj_to_load  # Loaded object