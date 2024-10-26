# Imports
from libs.pre_process import scaler_data  # pre_process.py
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle

# Default SEED
RANDOM_STATE = 42
# Default path
DATA_PATH = "data/"
BIN_PATH = "bin/"


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


def split_X_y(data, target_column="is_target", columns_to_drop=None):
    # Columns to drop
    columns_to_drop = columns_to_drop if columns_to_drop else []
    # DF with the feautures
    # Target column is added here to delete in X
    # DataFrame X (features)
    X = data.drop(columns_to_drop + [target_column], axis=1)
    # y (labels)
    y = data[target_column]

    return X, y  # Matrix X and vector y


def split_train_test(df, time_column="", target_column="is_target", test_size=0.20, random_state=RANDOM_STATE):
    # Copy original DataFrame
    train_df = df.copy()
    
    if time_column:  # time_column not empty
        # Sort DataFrame by time column
        train_df = train_df.sort_values(by=[time_column]).reset_index(drop=True)
        # Build test DataFrame
        test_df = pd.DataFrame(columns=train_df.columns)
    
        for i in np.sort(pd.unique(train_df[target_column])):  # For each class
            temp_df = train_df[train_df[target_column] == i]  # Select only data from a class
            # Obtain a percentage of data (end of DataFrame)
            temp_df = temp_df.tail(round(len(temp_df) * test_size))
            # Drop data obtained from the train DataFrame
            train_df.drop(index=temp_df.index, inplace=True)
            # Add data in test DataFrame
            test_df = pd.concat([test_df, temp_df])
    
        # Sort DataFrame by time column (again)
        # train_df is train/validation DataFrame
        train_df = train_df.sort_values(by=[time_column]).reset_index(drop=True)
        test_df = test_df.sort_values(by=[time_column]).reset_index(drop=True)

    else:
        # Split into X and y
        X, y = split_X_y(train_df, target_column)
        # Split into stratified train and test data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y)
    
        # Add y back into X
        X_train[target_column] = y_train
        X_test[target_column] = y_test
        # Reset index from previous data, renaming dataframes too
        train_df = X_train.reset_index(drop=True)
        test_df = X_test.reset_index(drop=True)
    
    # Return train_df and test_df
    return train_df, test_df


def save_object(obj_to_save, filename):
    # Save .pkl object
    pickle.dump(obj_to_save, open(f"{filename}.pkl", "wb"))


def load_object(filename):
    # Load .pkl object
    obj_to_load = pickle.load(open(f"{filename}.pkl", "rb"))

    return obj_to_load  # Loaded object


def load_data(data_path, target_column="is_target", random_state=RANDOM_STATE,
              test_size=0.2, columns_to_drop=["DateTime"], standardize=True, scaler_filename="scaler_obj"):
    # Load dataset
    df = pd.read_csv(data_path)
    # Split data into 80% for training and 20% for test (default)
    train_df, test_df = split_train_test(df, target_column=target_column,
                                         random_state=random_state, test_size=test_size)
    # Split training and test data into X and y
    X_train, y_train = split_X_y(train_df, target_column, columns_to_drop)
    X_test, y_test = split_X_y(test_df, target_column, columns_to_drop)

    if standardize:  # Standardize data
        # Load scale object...
        scaler = load_object(f"{BIN_PATH}{scaler_filename}")
        # Scale training and test data
        X_train, _ = scaler_data(X_train, scaler)
        X_test, _ = scaler_data(X_test, scaler)

    # Return train and test data
    return X_train, y_train, X_test, y_test