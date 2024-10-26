# Imports
from sklearn.preprocessing import MinMaxScaler
import ast
import pandas as pd

# Default SEED
RANDOM_STATE = 42


def categorical_encoder(df, suffixes, prefix):
    # Add new columns for each unique value in 'suffixes'
    for suffix in suffixes:
        column_name = f"{prefix}_{suffix}"
        # Initialize new column with zero (0)
        df[column_name] = 0
        # Set the value to 1 where the original column matches the suffix
        df.loc[df[prefix] == suffix, column_name] = 1

    # Drop original categorical column
    df.drop(columns=prefix, inplace=True)
    # DataFrame after encoder
    return df


def preprocess_flight_data(df, selected_cols=None):
    # Convert string column 'Sensors' to dictionary
    df["Sensors"] = df["Sensors"].apply(ast.literal_eval)
    # Add new 'Sensors' columns to the flight DataFrame
    df = df.join(pd.json_normalize(df["Sensors"])).drop("Sensors", axis=1)
    
    # Selected columns
    selected_cols = selected_cols if selected_cols else ["Timestamp",
        "SpeedChanged_speedX", "SpeedChanged_speedY", "SpeedChanged_speedZ",
        "AltitudeChanged_altitude",
        "AttitudeChanged_roll", "AttitudeChanged_pitch", "AttitudeChanged_yaw",
        "GpsLocationChanged_latitude", "GpsLocationChanged_longitude", "GpsLocationChanged_altitude",
        "GpsLocationChanged_latitude_accuracy", "GpsLocationChanged_longitude_accuracy", "GpsLocationChanged_altitude_accuracy",
        "HomeChanged_latitude", "HomeChanged_longitude", "HomeChanged_altitude",
        "HomeTypeAvailabilityChanged_type",
        "FlyingStateChanged_state", 
        "moveByEnd_dX", "moveByEnd_dY", "moveByEnd_dZ", "moveByEnd_dPsi", 
        "WifiSignalChanged_rssi",
        "BatteryStateChanged_percent"]
    # Only selected columns in DataFrame
    df = df[selected_cols]
    
    # Replacing NaN data
    # Categorial data
    df["HomeTypeAvailabilityChanged_type"] = df["HomeTypeAvailabilityChanged_type"].fillna("UNKNOWN")
    df["FlyingStateChanged_state"] = df["FlyingStateChanged_state"].fillna("unknown")
    df.fillna(0, inplace=True)  # Numerical data
    
    # Convert Timestamp to Datetime
    # Selection of specific flight times
    df["DateTime"] = pd.to_datetime(df["Timestamp"], unit="s")

    ### Categorical Encoder ###
    # FlyingStateChanged_state
    df = categorical_encoder(df, [
        "unknown",
        "landed",
        "takingoff",
        "hovering",
        "flying",
        "landing",
        "emergency",
        "usertakeoff",
        "motor_ramping",
        "motor_emergency_landing"], "FlyingStateChanged_state")
    
    # HomeTypeAvailabilityChanged_type
    df = categorical_encoder(df, [
        "UNKNOWN",
        "TAKEOFF",
        "PILOT",
        "FIRST_FIX",
        "FOLLOWEE"], "HomeTypeAvailabilityChanged_type")
    ### --- ###
    
    df.drop(columns="Timestamp", inplace=True)  # Drop non-useful columns. DateTime replaces Timestamp
    # DateTime as first column
    df = df[["DateTime"] + [col for col in df.columns if col != "DateTime"]]

    # Categorical columns
    categorical_cols = [col for col in df.columns if any(col.startswith(suffix) for suffix in [
        "FlyingStateChanged_state_", "HomeTypeAvailabilityChanged_type_"])]
    # Convert categorical columns to object
    df[categorical_cols] = df[categorical_cols].astype(object)
    # Numerical columns
    numerical_cols = sorted(list(set(df.drop(columns=["DateTime"]).columns) - set(categorical_cols)))
    
    return df, categorical_cols, numerical_cols  # DataFrame after preprocessing, and categorical & numerical columns


def preprocess_network_data(df):
    # Remove blank in column names and in lower case
    # Also, replace blank in the middle of the string with underscore
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    # Sort DataFrame by 'Time' column
    df = df.sort_values(by=["time"]).reset_index(drop=True)
    # Drop 'id' and 'is_ch' column
    df.drop(columns=["id", "is_ch"], inplace=True)
    # Rename class column
    df.rename(columns={"attack_type": "is_target"}, inplace=True)
    # Remove TDMA schedule attack
    df = df[df["is_target"] != "TDMA"].reset_index(drop=True)
    # Convert classes to numeric
    df["is_target"] = df["is_target"].map({
        "Normal": 0, "Grayhole": 1, "Blackhole": 2, "Flooding": 3}.get)

    # Categorical columns
    categorical_cols = [column for column in df.columns if df[column].isin([0, 1]).all()]
    # Numerical columns
    numerical_cols = sorted(list(set(df.drop(columns=["time", "is_target"]).columns) - set(categorical_cols)))

    return df, categorical_cols, numerical_cols  # DataFrame after preprocessing, and categorical & numerical columns


def drop_columns(df, numerical_cols, features, columns_to_drop):
    # Drop specified columns from the DataFrame
    df.drop(columns=columns_to_drop, inplace=True)
    # From lists now...
    numerical_cols = [x for x in numerical_cols if x not in columns_to_drop]
    features = [x for x in features if x not in columns_to_drop]

    return df, numerical_cols, features  # DataFrame after dropping columns


def scaler_data(data, scaler=None):
    if scaler is None:  # Unspecified scaler object
        scaler = MinMaxScaler()  # Initialize scaler
        scaler.fit(data)  # Fit scaler
        
    data_norm = scaler.transform(data)  # Scaling data...

    # Scaled DataFrame + scaler object
    return pd.DataFrame(data_norm, columns=data.columns), scaler