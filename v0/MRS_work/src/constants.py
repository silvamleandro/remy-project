# constants.py

# Project name
PROJECT_NAME = "REMY-MRS"

# Desired topics and their types
TOPICS_TYPES = {
    "mavros/gpsstatus/gps1/raw": "gps",
    "mavros/global_position/compass_hdg": "gps",
    "mavros/local_position/pose": "pos_orient",
    "mavros/local_position/velocity_local": "pos_orient",
    "mavros/altitude": "pos_orient",
    "mavros/imu/data_raw": "imu",
    "mavros/imu/mag": "imu",
    "mavros/imu/static_pressure": "imu",
    "mavros/imu/temperature_imu": "imu",
    "mrs_uav_status/uav_status": "sys_status"
}

# Default seed
RANDOM_STATE = 42
