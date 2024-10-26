# Imports
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score
)
import argparse
import flwr as fl
import numpy as np
import os
import sys
import warnings

# libs
sys.path.append(os.path.abspath(os.path.abspath(
    os.path.join(os.path.expanduser("~") + "/remy-project/"))))  # path
from libs.utils import load_data
from libs.fl_autoencoder import (
    create_model,
    reconstruction_loss,
    distance_calculation
)


# Ignore warnings
warnings.filterwarnings("ignore")

verbose = 0 # Show metrics results or not


def evaluate_learning(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)  # Accuracy
    recall = recall_score(y_true, y_pred)  # Recall
    precision = precision_score(y_true, y_pred)  # Precision
    f1 = f1_score(y_true, y_pred)  # F1-score
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()  # Confusion matrix
    missrate = fn / (fn + tp)  # Miss rate
    fallout = fp / (fp + tn)  # Fall-out
    auc = roc_auc_score(y_true, y_pred)  # ROC AUC

    # Evaluation metrics
    return accuracy, recall, precision, f1, missrate, fallout, auc


class FlwrClient(fl.client.NumPyClient):
    # Implement flower client extending class NumPyClient
    # This class serialize and de-serialize the weights into NumPy ndarray

    def __init__(self, data_path):
        self.data_path = data_path
        self.X_train, self.y_train, self.X_test, self.y_test = load_data(self.data_path)

        input_dim = self.X_train.shape[1] # Number of predictor variables
        # Create autoencoder model
        self.model = create_model(input_dim)

        self.loss = 0
        self.threshold_normal = 0  # Threshold calculated on normal samples from train data during evaluate
        self.threshold_abnormal = 0  # Threshold calculated on abnormal samples from train data during evaluate

        train_data = self.X_train[self.y_train == 0] # Only normal samples
        # Separate 90% of the indexes for validation
        idx = int(train_data.shape[0] * 0.90)
        self.val_data = train_data[idx:]  # Hold-out validation set for threshold calculation
        self.train_data = train_data[:idx]  # Reduced x_train (with out val_data)
        # Abnormal data from train set
        # Used only for threshold estimation
        self.abnormal_data = self.X_train[self.y_train != 0]

    def get_parameters(self):
        # Return local model parameters
        return self.model.get_weights() # get_weights from Keras returns the weights as ndarray

    def set_parameters(self, parameters):
        # Server sets model parameters from a list of NumPy ndarrays (Optional)
        self.model.set_weights(parameters) # set_weights on local model (similar to get_weights)

    def fit(self, parameters, config):
        # Receive parameters from the server and use them to train on local data (locally)
        self.set_parameters(parameters)
        # Fit autoencoder
        history = self.model.fit(
            self.train_data,
            self.train_data,
            verbose=0,
            batch_size=config["batch_size"],
            shuffle=True,
            epochs=config["num_epochs"] # Single epoch on local data
        )

        # History loss
        self.loss = history.history["loss"][-1]

        # return the refined model parameters with get_weights, local data length, and history loss
        # len(x_train) is a useful information for FL, analogous to weights of contribution of each client
        return self.get_parameters(), len(self.train_data), {"loss": history.history["loss"][-1], }

    def evaluate(self, parameters, config):
        # Evaluates the model on local data (locally)
        self.set_parameters(parameters)

        # Eval model on hold-out validation data
        val_inference = self.model.predict(self.val_data, verbose=0)
        # Eval model on abnormal data
        abnormal_inference = self.model.predict(self.abnormal_data, verbose=0)
        # Calculate reconstruction loss for validation and abnormal data
        val_losses = reconstruction_loss(self.val_data, val_inference)
        abnormal_losses = reconstruction_loss(self.abnormal_data, abnormal_inference)

        # Threshold calculation
        self.threshold_normal = np.mean(val_losses)
        self.threshold_abnormal = np.mean(abnormal_losses)
        # Show mean validation loss for normal and abnormal data (threshold)
        if verbose == 1:
            print("\nMean Validation Loss (Normal): {} | (Abnormal): {}".format(self.threshold_normal,
                                                                                     self.threshold_abnormal))

        # Test set evaluation
        inference = self.model.predict(self.X_test, verbose=0)
        losses = reconstruction_loss(self.X_test, inference)

        # Threshold criteria
        test_eval = distance_calculation(losses, self.threshold_normal, self.threshold_abnormal)
        # Evaluate model learning with test data
        accuracy, recall, precision, f1, missrate, fallout, auc = evaluate_learning(self.y_test, test_eval)
        # Save metrics to a dictionary
        metrics_dict = {"accuracy": accuracy,
                        "recall": recall,
                        "precision": precision,
                        "f1_score": f1,
                        "missrate": missrate,
                        "fallout": fallout,
                        "auc": auc}

        # Show metrics on test data
        if verbose == 1:
            print("\nThreshold: {} \n\nMetrics: {} \n\nMean Abnormal Loss: {} \nMean Normal Loss: {}\n".format(
                self.threshold_normal, metrics_dict, np.mean(losses[self.y_test == 1]), np.mean(losses[self.y_test == 0])))

        # Return loss, test data length and metrics dictionary
        return float(self.loss), len(self.X_test), metrics_dict


def main():
    # Parse command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    # Start Flower client		
    client = FlwrClient(args.data_path).to_client()
    # Start NumPy client
    fl.client.start_client(server_address="0.0.0.0:8080", client=client)

if __name__ == "__main__":
    main()
