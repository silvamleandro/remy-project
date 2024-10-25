# Imports
from libs.balance_data import BalanceData
from libs.pre_process import preprocess_network_data  # pre_process.py
from libs.utils import split_train_test, split_X_y  # utils.py
from ydata_synthetic.synthesizers.regular import RegularSynthesizer
import pandas as pd
import warnings


# Ignore warnings
warnings.filterwarnings("ignore")

# Default SEED
RANDOM_STATE = 42
# Default path
DATA_PATH = "data/"
# Default path
BIN_PATH = "bin/"


# Load data
network_df = pd.read_csv(f"{DATA_PATH}raw/WSN_DS.csv")

# Pre-processing network data...
network_df, _, _ = preprocess_network_data(network_df)
# Split into UAV 1 and UAV 2 data
uav_1_df, uav_2_df = split_train_test(network_df, time_column="time", target_column="is_target", test_size=0.50)

# Selected features in Boruta
boruta_features = ["join_s", "who_ch", "data_s", "expaned_energy", "dist_ch_to_bs", "send_code", "rank", "adv_r"]
# Selecting...
uav_1_df = uav_1_df[boruta_features + ["is_target"]]
uav_2_df = uav_2_df[boruta_features + ["is_target"]]

for i, df in enumerate([uav_1_df, uav_2_df]):
    # Split into X and y
    X, y = split_X_y(df, "is_target", [])
    # Convert y to integer
    y = y.astype(int)
    # Create a BalanceData object for resampling
    balance = BalanceData(X, y)
    # Apply RUS to reduce majority class to sum of minority classes
    balance.rus(params={"random_state": RANDOM_STATE, "sampling_strategy": dict(sorted(
        {label: (y != 0).sum() if label == 0 else (y == label).sum() for label in y.unique()}.items()))})
    X_res, y_res, _ = balance.fit_resample()

    # Generate synthetic data for each class, except the majority class
    for label in sorted(y_res[y_res != 0].unique()):
        synth = RegularSynthesizer.load(f"{BIN_PATH}resampling/CTGAN_c{label}_obj.pkl")

        # Generate new synthetic data
        n = int(round((y_res == label).sum() * 0.5))
        synth_data = synth.sample(n)
        X_res = pd.concat([X_res, synth_data], ignore_index=True)
        y_res = pd.concat([y_res, pd.Series(label, index=range(n))], ignore_index=True)

    X_res["is_target"] = y_res
    # Save data after pre-processing
    df.to_csv(f"{DATA_PATH}uav_{i + 1}.csv", index=False)