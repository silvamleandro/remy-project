# Imports
from logging import INFO
from typing import Dict
import argparse
import flwr as fl
import os
import sys
import warnings

# libs
sys.path.append(os.path.abspath(os.path.abspath(
    os.path.join(os.path.expanduser("~") + "/remy-project/"))))  # path
from libs.utils import load_data
from libs.fl_autoencoder import create_model


# Ignore warnings
warnings.filterwarnings("ignore")


def fit_config(rnd: int) -> Dict:
    config = {
        "epoch_global": str(rnd),
        "num_epochs": 50,
        "batch_size": 32
    }
    # Configuration for each training
    return config


def average_metrics(metrics):
    # Get local metrics
    accuracies = [metric["accuracy"] for _, metric in metrics]
    recalls = [metric["recall"] for _, metric in metrics]
    precisions = [metric["precision"] for _, metric in metrics]
    f1s = [metric["f1_score"] for _, metric in metrics]
    missrates = [metric["missrate"] for _, metric in metrics]
    fallouts = [metric["fallout"] for _, metric in metrics]
    aucs = [metric["auc"] for _, metric in metrics]

    # Calculate global metrics
    accuracies = sum(accuracies) / len(accuracies)
    recalls = sum(recalls) / len(recalls)
    precisions = sum(precisions) / len(precisions)
    f1s = sum(f1s) / len(f1s)
    missrates = sum(missrates) / len(missrates)
    fallouts = sum(fallouts) / len(fallouts)
    aucs = sum(aucs) / len(aucs)

    # Metrics in dictionary
    return {"accuracy": round(accuracies, 5),
            "recall": round(recalls, 5),
            "precision": round(precisions, 5),
            "f1_score": round(f1s, 5),
            "missrate": round(missrates, 5),
            "fallout": round(fallouts, 5),
            "auc": round(aucs, 5)}


def select_strategy(strategy_num, min_available_clients, parameters):
    # Select and create strategy
    if strategy_num == 1:  # FedAvg
        fl.common.logger.log(INFO, "########################## (1) FedAvg ##########################")
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1,
            fraction_evaluate=1,
            min_available_clients=min_available_clients,
            evaluate_metrics_aggregation_fn=average_metrics,
            on_fit_config_fn=fit_config,
            initial_parameters=fl.common.ndarrays_to_parameters(parameters)
        )
    elif strategy_num == 2:  # FedAvgM
        fl.common.logger.log(INFO, "########################## (2) FedAvgM ##########################")
        strategy = fl.server.strategy.FedAvgM(
            fraction_fit=1,
            fraction_evaluate=1,
            min_available_clients=min_available_clients,
            evaluate_metrics_aggregation_fn=average_metrics,
            on_fit_config_fn=fit_config,
            initial_parameters=fl.common.ndarrays_to_parameters(parameters),
            server_learning_rate=1.0,
            server_momentum=0.95
        )
    elif strategy_num == 3:  # FedAdagrad
        fl.common.logger.log(INFO, "########################## (3) FedAdagrad ##########################")
        strategy = fl.server.strategy.FedAdagrad(
            fraction_fit=1,
            fraction_evaluate=1,
            min_available_clients=min_available_clients,
            evaluate_metrics_aggregation_fn=average_metrics,
            on_fit_config_fn=fit_config,
            initial_parameters=fl.common.ndarrays_to_parameters(parameters),
            eta=1e-1,
            eta_l=1e-1,
            tau = 1e-9
        )
    elif strategy_num == 4:  # FedAdam
        fl.common.logger.log(INFO, "########################## (4) FedAdam ##########################")
        strategy = fl.server.strategy.FedAdam(
            fraction_fit=1,
            fraction_evaluate=1,
            min_available_clients=min_available_clients,
            evaluate_metrics_aggregation_fn=average_metrics,
            on_fit_config_fn=fit_config,
            initial_parameters=fl.common.ndarrays_to_parameters(parameters),
            eta=1e-1,
            eta_l=1e-1,
            beta_1=1e-1,
            beta_2=1e-1,
            tau = 1e-9
        )
    elif strategy_num == 5:  # FedYogi
        fl.common.logger.log(INFO, "########################## (5) FedYogi ##########################")
        strategy = fl.server.strategy.FedYogi(
            fraction_fit=1,
            fraction_evaluate=1,
            min_available_clients=min_available_clients,
            evaluate_metrics_aggregation_fn=average_metrics,
            on_fit_config_fn=fit_config,
            initial_parameters=fl.common.ndarrays_to_parameters(parameters),
            eta=1e-1,
            eta_l=1e-1,
            beta_1=1e-1,
            beta_2=1e-1,
            tau = 1e-9
        )

    return strategy  # Select strategy


def main():
    # Parse command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--strategy_num", type=str, required=False)
    args = parser.parse_args()

    # Initializing a generic model to get its parameters
    X_train, _, _, _ = load_data(args.data_path)
    
    parameters = create_model(X_train.shape[1]).get_weights()  # Global autoencoder
    del X_train  # Delete X_train

    try:  # Select and create strategy
        strategy = select_strategy(strategy_num=int(args.strategy_num), min_available_clients=2, parameters=parameters)
    except UnboundLocalError: strategy = select_strategy(strategy_num=1, min_available_clients=2, parameters=parameters)
    
    # Start Flower server
    fl.server.start_server(server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy
    )

if __name__ == "__main__":
    main()
