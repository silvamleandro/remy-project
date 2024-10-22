# Imports
from imblearn.over_sampling import SMOTENC, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from libs.utils import save_object  # utils.py
from ydata_synthetic.synthesizers.regular import RegularSynthesizer
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
import pandas as pd

# Default SEED
RANDOM_STATE = 42
# Default path
BIN_PATH = "bin/"


class BalanceData:
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame):
        self.X = X
        self.y = y

    def rus(self, params={}):  # Random Undersampling (RUS)
        # RUS Parameters
        self.model = RandomUnderSampler(**params)

    def ros(self, params={}):  # Random Oversampling (ROS)
        # ROS Parameters
        self.model = RandomOverSampler(**params)
        
    def smotenc(self, params={}):  # Synthetic Minority Over-sampling Technique for Nominal and Continuous (SMOTENC)
        # SMOTENC Parameters
        self.model = SMOTENC(**params)

    def fit_resample(self):
        X_res, y_res = self.model.fit_resample(self.X, self.y)
        # Reduced X, y + Data Resampling Object
        return X_res.reset_index(drop=True), y_res.reset_index(drop=True), self.model


def resampling_data(X_train, y_train, numerical_cols, categorical_cols, epochs=500,
                 strategies=["ONLY_RUS", "ROS", "SMOTENC", "CTGAN"], object_to_save=False):
    # Dictionary to store the resampled data for each resampling strategy
    resampled = {key: {"X": pd.DataFrame(), "y": pd.Series()}
                 for key in strategies}

    ### Random Undersampling (RUS)
    print("\n>> ONLY_RUS: ...")
    # Create a BalanceData object for resampling
    balance = BalanceData(X_train, y_train)
    
    # Apply RUS to reduce majority class by 50%
    balance.rus(params={"random_state": RANDOM_STATE, "sampling_strategy": dict(sorted(
        {label: int(y_train.value_counts()[label] * (0.50 if label == 0 else 1)) for label in y_train.unique()}.items()))})
    resampled["ONLY_RUS"]["X"], resampled["ONLY_RUS"]["y"], resampling_obj = balance.fit_resample()
    if object_to_save: save_object(resampling_obj, f"{BIN_PATH}resampling/RUS_obj")  # Optionally save the resampling object
    print(">> ONLY_RUS: Done!\n")

    # Define new sampling strategy based on RUS results
    sampling_strategy = dict(sorted({label: int(resampled["ONLY_RUS"]["y"].value_counts()[label] * (1 if label == 0 else 1.5))
                                     for label in resampled["ONLY_RUS"]["y"].unique()}.items()))
    
    ### Random Oversampling (ROS)
    if "ROS" in strategies:
        print("\n>> ROS: ...")
        # Initialize BalanceData with undersampled data
        balance = BalanceData(resampled["ONLY_RUS"]["X"], resampled["ONLY_RUS"]["y"])
        # Apply ROS
        balance.ros(params={"random_state": RANDOM_STATE, "sampling_strategy": sampling_strategy})
        resampled["ROS"]["X"], resampled["ROS"]["y"], resampling_obj = balance.fit_resample()
        if object_to_save: save_object(resampling_obj, f"{BIN_PATH}resampling/ROS_obj")  # Optionally save the resampling object
        print(">> ROS: Done!\n")

    ### Synthetic Minority Over-sampling Technique for Nominal and Continuous (SMOTENC)
    if "SMOTENC" in strategies:
        print("\n>> SMOTENC: ...")
        # Initialize BalanceData with undersampled data
        balance = BalanceData(resampled["ONLY_RUS"]["X"], resampled["ONLY_RUS"]["y"])
        # Apply SMOTENC
        balance.smotenc(params={"random_state": RANDOM_STATE, "sampling_strategy": sampling_strategy,
                                "categorical_features": categorical_cols})
        resampled["SMOTENC"]["X"], resampled["SMOTENC"]["y"], resampling_obj = balance.fit_resample()
        if object_to_save: save_object(resampling_obj, f"{BIN_PATH}resampling/SMOTENC_obj")  # Optionally save the resampling object
        print(">> SMOTENC: Done!\n")

    ### Conditional Tabular Generative Adversarial Network (CTGAN)
    if "CTGAN" in strategies:
        print("\n>> CTGAN: ...")
        
        resampled["CTGAN"]["X"] = pd.DataFrame()
        resampled["CTGAN"]["y"] = pd.Series()

        # Defining the training parameters
        ctgan_args = ModelParameters(batch_size=500, lr=2e-4, betas=(0.5, 0.9))
        train_args = TrainParameters(epochs=epochs + 1)

        # Generate synthetic data for each class, except the majority class
        for label in sorted(y_train[y_train != 0].unique()):
            print(f"\n> Class {label}:")
            synth = RegularSynthesizer(modelname="ctgan", model_parameters=ctgan_args)
            synth.fit(data=X_train[y_train == label], train_arguments=train_args,
                      num_cols=numerical_cols, cat_cols=categorical_cols)
            if object_to_save: synth.save(f"{BIN_PATH}resampling/CTGAN_c{label}_obj.pkl")  # Optionally save the resampling object

            # Generate new synthetic data
            n = int(round((y_train == label).sum() * 0.5))
            synth_data = synth.sample(n)
            resampled["CTGAN"]["X"] = pd.concat([resampled["CTGAN"]["X"], synth_data], ignore_index=True)
            resampled["CTGAN"]["y"] = pd.concat([resampled["CTGAN"]["y"], pd.Series(label, index=range(n))], ignore_index=True)

        # Combine results from RUS and CTGAN
        resampled["CTGAN"]["X"] = pd.concat([resampled["ONLY_RUS"]["X"], resampled["CTGAN"]["X"]], ignore_index=True)
        resampled["CTGAN"]["y"] = pd.concat([resampled["ONLY_RUS"]["y"], resampled["CTGAN"]["y"]], ignore_index=True)
        print("\n>> CTGAN: Done!\n")
        
    # Dictionary with the balanced data for each strategy
    return resampled