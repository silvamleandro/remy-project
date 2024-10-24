# Imports
from libs.utils import load_data
from logging import INFO
from sklearn.metrics import precision_recall_fscore_support
import argparse
import flwr as fl
import numpy as np
import xgboost as xgb
import warnings


# Ignore warnings
warnings.filterwarnings("ignore")

# Define arguments parser for the client/partition ID.
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

# Load data
fl.common.logger.log(INFO, "Loading data...")
X_train, y_train, X_test, y_test = load_data(args.data_path, columns_to_drop=[], standardize=False)

# Reformat data to DMatrix for XGBoost
fl.common.logger.log(INFO, "Reformatting data...")
# Transform dataset to DMatrix format for XGBoost
train_dmatrix = xgb.DMatrix(X_train, label=y_train)
valid_dmatrix = xgb.DMatrix(X_test, label=y_test)

# Hyperparameters for XGBoost training
num_local_round = 1
params = {  # Update
    "objective": "multi:softprob",
    "eta": 0.1,  # Learning rate
    "num_class": y_train.nunique(),
    "max_depth": 8,
    "eval_metric": "auc",
    "nthread": 16,
    "num_parallel_tree": 1,
    "subsample": 1,
    "tree_method": "hist"}


# Define Flower client
class XgbClient(fl.client.Client):
    def __init__(self, train_dmatrix, valid_dmatrix, num_train, num_val, num_local_round, params):
        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix
        self.num_train = num_train
        self.num_val = num_val
        self.num_local_round = num_local_round
        self.params = params

    def get_parameters(self, ins: fl.common.GetParametersIns) -> fl.common.GetParametersRes:
        _ = (self, ins)
        return fl.common.GetParametersRes(status=fl.common.Status(code=fl.common.Code.OK, message="OK"),
                                          parameters=fl.common.Parameters(tensor_type="", tensors=[]))

    def _local_boost(self, bst_input):
        # Update trees based on local training data.
        for i in range(self.num_local_round):
            bst_input.update(self.train_dmatrix, bst_input.num_boosted_rounds())

        # Bagging: extract the last N=num_local_round trees for sever aggregation
        bst = bst_input[
            bst_input.num_boosted_rounds()
            - self.num_local_round : bst_input.num_boosted_rounds()]

        return bst

    def fit(self, ins: fl.common.FitIns) -> fl.common.FitRes:
        global_round = int(ins.config["global_round"])
        if global_round == 1:
            # First round local training
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
                evals=[(self.valid_dmatrix, "validate"), (self.train_dmatrix, "train")])
        else:
            bst = xgb.Booster(params=self.params)
            for item in ins.parameters.tensors:
                global_model = bytearray(item)

            # Load global model into booster
            bst.load_model(global_model)
            # Local training
            bst = self._local_boost(bst)

        # Save model
        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)

        return fl.common.FitRes(status=fl.common.Status(code=fl.common.Code.OK, message="OK"),
                                parameters=fl.common.Parameters(tensor_type="", tensors=[local_model_bytes]),
                                num_examples=self.num_train, metrics={})

    def evaluate(self, ins: fl.common.EvaluateIns) -> fl.common.EvaluateRes:
        # Load global model
        bst = xgb.Booster(params=self.params)
        for para in ins.parameters.tensors:
            para_b = bytearray(para)
        bst.load_model(para_b)

        # Make predictions
        preds = bst.predict(self.valid_dmatrix)
        predicted_labels = np.argmax(preds, axis=1)

        # Run evaluation
        eval_results = bst.eval_set(
            evals=[(self.valid_dmatrix, "valid")],
            iteration=bst.num_boosted_rounds() - 1)
        
        auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)

        # Calculate recall and F1-score
        true_labels = self.valid_dmatrix.get_label()
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average="macro")

        return fl.common.EvaluateRes(status=fl.common.Status(code=fl.common.Code.OK, message="OK"),
                                     loss=0.0, num_examples=self.num_val, metrics=
                                     {"AUC": auc, "Precision": precision, "Recall": recall, "F1": f1})


# Start Flower client
fl.client.start_client(server_address="127.0.0.1:9030",
                       client=XgbClient(train_dmatrix, valid_dmatrix,len(X_train), len(X_test),
                                        num_local_round, params).to_client())
