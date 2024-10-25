# utils.py

# Imports
from datetime import timedelta
from pathlib import Path
from rosbags.dataframe import get_dataframe
from rosbags.highlevel import AnyReader
from sklearn.preprocessing import MinMaxScaler

import constants # constants.py
import joblib
import math
import numpy as np
import pandas as pd


def columns_by_topic(topic):
    '''
        :Param:
            topic: topic where columns are selected
            
        :Return:
            List of columns selected in the topic

        :Description:
            Select which columns are used in the topic
            Note: The focus of this work is the detection of anomalies in sensors.
            (Columns vary depending on the problem)
    '''

    # Column list (empty)
    columns = []

    # Select the topic
    if topic == "mavros/gpsstatus/gps1/raw" or topic == 0: # mavros/gpsstatus/gps1/raw (0)
        columns.extend(["lat", "lon", "alt", "eph", "epv", "vel", "cog"])
    
    elif topic ==  "mavros/global_position/compass_hdg" or topic == 1: # mavros/global_position/compass_hdg (1)
        columns.extend(["data"])
    
    elif topic == "mavros/local_position/pose" or topic == 2: # mavros/local_position/pose (2)
        columns.extend(["pose.position.x", "pose.position.y", "pose.position.z",
                        "pose.orientation.w", "pose.orientation.x", "pose.orientation.y", "pose.orientation.z"])
    
    elif topic == "mavros/local_position/velocity_local" or topic == 3: # mavros/local_position/velocity_local (3)
        columns.extend(["twist.linear.x", "twist.linear.y", "twist.linear.z",
                        "twist.angular.x", "twist.angular.y", "twist.angular.z"])
    
    elif topic == "mavros/altitude" or topic == 4: # mavros/altitude (4)
        columns.extend(["relative", "local", "amsl"])
    
    elif topic == "mavros/imu/data_raw" or topic == 5: # mavros/imu/data_raw (5)
        columns.extend(["angular_velocity.x", "angular_velocity.y", "angular_velocity.z",
                        "linear_acceleration.x", "linear_acceleration.y", "linear_acceleration.z"])
    
    elif topic == "mavros/imu/mag" or topic == 6: # mavros/imu/mag (6)
        columns.extend(["magnetic_field.x", "magnetic_field.y", "magnetic_field.z"])
    
    elif topic == "mavros/imu/static_pressure" or topic == 7: # mavros/imu/static_pressure (7)
        columns.extend(["fluid_pressure"])
    
    elif topic == "mavros/imu/temperature_imu" or topic == 8: # mavros/imu/temperature_imu (8)
        columns.extend(["temperature"])
    
    elif topic == "mrs_uav_status/uav_status" or topic == 9: # mrs_uav_status/uav_status (9)
        columns.extend(["cpu_load", "cpu_temperature", "free_ram", "battery_wh_drained", "thrust"])
    else: # Invalid topic
        print("Incorrect topic!")
        return [] # Empty list
        
    return columns # Selected columns



def euler_from_quaternion(w, x, y, z, angle_measure="rad"):
    '''
        :Param:
            w, x, y, z: quaternion
            angle_measure: angles in radians or degrees
            
        :Return:
            Roll, pitch and yaw after converting

        :Description:
            Convert a quaternion into Euler angles (roll, pitch, yaw)
            Note: By default, the function returns in radians
    '''

    # (1) Roll
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    # (2) Pitch
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    # (3) Yaw
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    if angle_measure == "deg": # Convert to degrees
        roll, pitch, yaw = math.degrees(roll), math.degrees(pitch), math.degrees(yaw)
                
    return roll, pitch, yaw # Euler angles



def get_dataframes(path, bag_name, uav_name, topics=list(constants.TOPICS_TYPES.keys()), angle_measure="rad"):
    '''
        :Param:
            path: Path where the .bag file is located
            bag_name: .bag file name
            uav_name: name of the Unmanned Aerial Vehicle (UAV) that is reading from the bag
            topics: topics that are selected from the bag
            angle_measure: angles in radians or degrees
            
        :Return:
            DataFrames dictionary, where each one corresponds to a topic

        :Description:
            Read a bag and return data from selected topics in DataFrames
    '''

    with AnyReader([Path(f"{path}/{bag_name}.bag")]) as reader: # Open the bag
        dfs = dict() # DataFrame dictionary with each topic

        for topic in topics: # Loop through each topic
            df = get_dataframe(reader, f"/{uav_name}/{topic}", columns_by_topic(topic))
            df.reset_index(inplace=True) # Reset index
            df = df.rename(columns = {"index":"Time"}) # Time as a DataFrame column
            dfs[topic] = df # Add to dictionary

    if "mavros/local_position/pose" in topics: # Add Euler angles columns
        # Vectorize the existing function
        v_euler_from_quaternion = np.vectorize(euler_from_quaternion)
        # Calculate Euler angles
        dfs["mavros/local_position/pose"][f"roll_{angle_measure}"],\
            dfs["mavros/local_position/pose"][f"pitch_{angle_measure}"],\
                dfs["mavros/local_position/pose"][f"yaw_{angle_measure}"] = v_euler_from_quaternion(
                    dfs["mavros/local_position/pose"]["pose.orientation.w"].to_numpy(),
                    dfs["mavros/local_position/pose"]["pose.orientation.x"].to_numpy(),
                    dfs["mavros/local_position/pose"]["pose.orientation.y"].to_numpy(),
                    dfs["mavros/local_position/pose"]["pose.orientation.z"].to_numpy(),
                    angle_measure=angle_measure # Radians (rad) or Degrees (deg)
                )
    
    return dfs # DataFrame dictionary



def min_max_timestamp(dfs):
    '''
        :Param:
            dfs: DataFrame dictionary where timestamp column is compared
            
        :Return:
            The minimum and maximum timestamp found from the DataFrames

        :Description:
            Find the minimum and maximum timestamp of the set of DataFrames
    '''

    # Minimum and maximum timestamp of the first DataFrame
    first_key = list(dfs.keys())[0] # Get first key
    min = dfs[first_key][["Time"]].min()[0]
    max = dfs[first_key][["Time"]].max()[0]

    for key in dfs: # Loop through DataFrame dictionary
        df = dfs[key] # Get DataFrame based on key

        # Compare if any topic timestamp is less than min variable
        if df[["Time"]].min()[0] < min:
            min = df[["Time"]].min()[0] # Update min timestamp
        
        # Compare if any topic timestamp is greater than max variable
        if df[["Time"]].max()[0] > max:
            max = df[["Time"]].max()[0] # Update max timestamp
        
    # Return minimum and maximum timestamp
    return min, max



def concatenate_str_in_list(string, list):
    '''
        :Param:
            string: string that is concatenated
            list: target list
            
        :Return:
            List with concatenated string

        :Description:
            Concatenate a string at the beginning of list elements
    '''

    # List with the string concatenated to the beginning of all elements
    return [f"{string}.{x}" for x in list]



def merge_dataframes(dfs, time_window=500, topics_types=constants.TOPICS_TYPES):
    '''
        :Param:
            dfs: DataFrames that are merged
            time_window: time window (in milliseconds) that data is merged
            topics_types: selected topics and their types
            
        :Return:
            DataFrame merged with data from each topic

        :Description:
            Merge multiple DataFrames into a single one based on a time window
    '''

    columns = [] # Empty columns list

    for topic_type in topics_types: # Loop through each topic and type
        # Add columns to the list
        columns += concatenate_str_in_list(topics_types[topic_type], list(dfs[topic_type].columns))

    # No "Time" columns
    columns = [x for x in columns if "Time" not in x]
    columns.insert(0, "Time")

    # Minimum and maximum time
    min_time, max_time = min_max_timestamp(dfs)
    time_window = timedelta(milliseconds=time_window) # Convert time window
    timestamp = min_time # First timestamp is the minimum timestamp
    timestamps = [] # Empty timestamp list

    while timestamp < max_time: # Generate timestamps based on time window
        timestamps.append(timestamp) # Add timestamp to list
        timestamp += time_window # Sum timestamp with time window
    else: # timestamp >= max_timestamp
        timestamps.append(max_time) # Last timestamp is the maximum timestamp

    merge_df = pd.DataFrame(columns=columns) # Generate empty merge DataFrame
    merge_df["Time"] = timestamps # Add timestamps with time window

    for key in dfs: # Loop through DataFrame dictionary
        df = dfs[key] # Get DataFrame based on key
        values = [] # Empty values

        for i in range(len(timestamps)): # Number of timestamp in merge DataFrame
            try: # Merge the data using the mean
                mean_df = list(df[(df["Time"] >= merge_df["Time"][i]) & 
                                  (df["Time"] < merge_df["Time"][i + 1])].mean(numeric_only=True))
                values.append(mean_df) # Add in values
            except KeyError: # If it reaches the last index
                mean_df = list(df[df["Time"] >= merge_df["Time"][i]].mean(numeric_only=True))
                values.append(mean_df) # Add in values
        
        # Add new values to the merge DataFrame
        merge_df[concatenate_str_in_list(topics_types[key], df.columns)[1:]] = values

    # Column list without "Time" column
    columns_without_time = list(merge_df.columns)[1:]
    # Interpolation with the nearest method to fill NaN values
    merge_df[columns_without_time] = merge_df[columns_without_time].interpolate(method="nearest")
    # Drop all the NaN rows
    merge_df.dropna(inplace=True)
    # Return merge DataFrame
    return merge_df.reset_index(drop=True)



def normalize_data(df, unnorm_cols=["Time"], create_scaler=True, scaler_filename="scaler"):
    '''
        :Param:
            df: DataFrame that is normalized
            unnorm_cols: columns that are not normalized
            create_scaler: whether scaler file is generated or not
            scaler_filename: scaler file name 

        :Return:
            DataFrame after normalizing

        :Description:
            Normalize a DataFrame
    '''

    norm_df = df.drop(columns=unnorm_cols) # DataFrame that will be normalized

    if create_scaler: # Create a scaler file
        scaler = MinMaxScaler()
        # Compute the minimum and maximum to be used for scaling
        scaler.fit(norm_df)
        # Save the generated scaler
        joblib.dump(scaler, f"{str(Path.home())}/{constants.PROJECT_NAME}/{scaler_filename}.pkl")
    else: # Load scaler file
        scaler = joblib.load(f"{str(Path.home())}/{constants.PROJECT_NAME}/{scaler_filename}.pkl")
    
    # Normalized data
    norm_df = pd.DataFrame(scaler.transform(norm_df), columns=norm_df.columns)
    # Update DataFrame with normalized data
    df = df[['Time']].join(norm_df)
    # Return normalized DataFrame
    return df
