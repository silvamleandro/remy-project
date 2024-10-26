# Imports
from tensorflow.keras import layers
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# Default SEED
RANDOM_STATE = 42


# Autoencoder
class Autoencoder:
    def __init__(self, X_train: pd.DataFrame, y_train: pd.DataFrame, perc_train=0.90):
        self.X_train = X_train
        self.y_train = y_train
        # Percentage of training data. Rest is for validation! 
        self.perc_train = perc_train
        self.train_data = pd.DataFrame(),
        self.val_data = pd.DataFrame(),
        self.abnormal_data = pd.DataFrame()
        self._split_train_data()  # Split training data
        self.model = tf.keras.Sequential()  # Empty Sequential Model
        self.threshold_normal = 0.0
        self.threshold_abnormal = 0.0

    def _split_train_data(self):
        # Only Normal Flight
        train_data = self.X_train[self.y_train == 0]
        # (100 - perc_train)% of Normal indexes for validation
        idx = int(train_data.shape[0] * self.perc_train)
        # Hold-out between validation and training
        self.val_data = train_data[idx:]
        self.train_data = train_data[:idx]  # Reduced X_train
        # Abnormal Flight samples (only for threshold estimation)
        self.abnormal_data = self.X_train[self.y_train != 0]

    def create_model(self):
        # Number of predictor variables
        input_dim = self.train_data.shape[1]
        # Input layer
        input = tf.keras.layers.Input(shape=(input_dim,))

        # Encoder layer
        encoder = tf.keras.Sequential([
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(8, activation="relu"),
            # Bottleneck
            layers.Dense(4, activation="relu")])(input)

        # Decoder layer
        decoder = tf.keras.Sequential([
            layers.Dense(8, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(input_dim, activation="sigmoid")])(encoder)

        # Instantiate autoencoder
        self.model = tf.keras.Model(inputs=input, outputs=decoder)
        # Compile...
        self.model.compile(optimizer="adam", loss="mean_squared_error")
        # input_dim -> 32 -> 16 -> 8 -> 4 -> 8 -> 16 -> 32 -> input_dim

    def __reconstruction_loss(self, x, x_hat):
        return np.mean(abs(x - x_hat), axis=1)  # Mean Absolute Error (MAE)

    def __get_thresholds(self):
        # Evaluate model on validation data
        val_inference = self.model.predict(self.val_data, verbose=0)
        # On abnormal data
        abnormal_inference = self.model.predict(self.abnormal_data, verbose=0)

        # Calculate reconstruction loss for validation and abnormal data
        val_losses = self.__reconstruction_loss(self.val_data, val_inference)
        abnormal_losses = self.__reconstruction_loss(self.abnormal_data, abnormal_inference)

        # Calculate each threshold
        self.threshold_normal = np.mean(val_losses)
        self.threshold_abnormal = np.mean(abnormal_losses)

        # Mean Validation Loss for normal and abnormal data (threshold)
        print(
            f"\nMean Validation Loss (Normal): {self.threshold_normal:.6f} | (Abnormal): {self.threshold_abnormal:.6f}")

    def fit_model(self, epochs=100, batch_size=32, shuffle=True, verbose=0):
        # Train autoencoder
        history = self.model.fit(self.train_data, self.train_data,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 verbose=verbose)

        # Calculate thresholds for normal and abnormal samples
        self.__get_thresholds()

        # Trained model, thresholds (normal and abnormal) and loss history
        return self.model, (self.threshold_normal, self.threshold_abnormal), history.history["loss"][-1]

    def __distance_calculation(self, losses):
        # Values that will be predicted
        preds = np.zeros(len(losses))

        # For each loss, calculate the distance to the closest threshold
        for i, loss in enumerate(losses):
            if abs(loss - self.threshold_normal) > abs(loss - self.threshold_abnormal):
                preds[i] = 1  # Abnormal
            else:
                preds[i] = 0  # Normal

        # Predicted values
        return preds

    def predict(self, X_test):
        # Evaluate test data
        inference = self.model.predict(X_test, verbose=0)
        losses = self.__reconstruction_loss(X_test, inference)
        # Classification by threshold
        return self.__distance_calculation(losses).astype(np.int64)

    def plot_reconstrunction_error(self, X_test):
        test_predictions = self.model.predict(
            X_test, verbose=0)  # Predict on test data

        # Calculate Mean Squared Error (MSE)
        mse = np.mean(np.power(X_test - test_predictions, 2), axis=1)
        # Generate error DataFrame
        error_df = pd.DataFrame({"Reconstruction_error": mse})

        _, ax = plt.subplots(figsize=(12, 8))  # Plot settings
        ax.plot(error_df.index, error_df.Reconstruction_error,
                marker="o", ms=3.5, linestyle="")  # Plot
        # "Normal Threshold" line
        ax.hlines(self.threshold_normal, ax.get_xlim()[0], ax.get_xlim()[
                  1], colors="g", zorder=100, label="Normal Threshold")
        # "Abnormal Threshold" line
        ax.hlines(self.threshold_abnormal, ax.get_xlim()[0], ax.get_xlim()[
                  1], colors="r", zorder=100, label="Abnormal Threshold")

        ax.legend()  # Legend
        plt.ylabel("Reconstruction Error")
        plt.xlabel("Data Index")
        plt.show()  # Show plot

        return error_df  # error DataFrame
    

# One-Class SVM
class OneClassSVMWrapper(OneClassSVM):
    def __init__(self, *, kernel="rbf", degree=3, gamma="scale", coef0=0.0, tol=1e-3,
                 nu=0.5, shrinking=True, cache_size=200, verbose=False, max_iter=-1):
        super().__init__(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            nu=nu,
            shrinking=shrinking,
            cache_size=cache_size,
            verbose=verbose,
            max_iter=max_iter
        )

    def predict(self, X):  # Output mapped to 0 and 1, instead of 1 and -1
        return np.fromiter(map({-1: 1, 1: 0}.get, super().predict(X)), dtype=int)


# Local Outlier Factor (LOF)
class LocalOutlierFactorWrapper(LocalOutlierFactor):
    def __init__(self, *, n_neighbors=20, algorithm="auto", leaf_size=30, metric="minkowski",
                 p=2, metric_params=None, contamination="auto", novelty=True, n_jobs=-1):
        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            contamination=contamination,
            novelty=novelty,
            n_jobs=n_jobs
        )

    def predict(self, X):  # Output mapped to 0 and 1, instead of 1 and -1
        return np.fromiter(map({-1: 1, 1: 0}.get, super().predict(X)), dtype=int)
    

# Isolation Forest
class IsolationForestWrapper(IsolationForest):
    def __init__(self, *, n_estimators=100, max_samples="auto", contamination="auto", max_features=1.0,
                 bootstrap=False, n_jobs=-1, random_state=RANDOM_STATE, verbose=0, warm_start=False):
        super().__init__(
            bootstrap=bootstrap,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            contamination=contamination
        )

    def predict(self, X):  # Output mapped to 0 and 1, instead of 1 and -1
        return np.fromiter(map({-1: 1, 1: 0}.get, super().predict(X)), dtype=int)