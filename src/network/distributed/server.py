# Imports
from flwr.server.strategy import FedXgbBagging
from typing import Dict
import flwr as fl
import warnings


# Ignore warnings
warnings.filterwarnings("ignore")

# FL experimental settings
pool_size = 2
num_rounds = 10
num_clients_per_round = 2
num_evaluate_clients = 2


def evaluate_metrics_aggregation(eval_metrics):
    # Aggregated metrics for evaluation
    total_num = sum([num for num, _ in eval_metrics])
    auc_aggregated = (sum([metrics["AUC"] * num for num, metrics in eval_metrics]) / total_num)
    precision_aggregated = (sum([metrics["Precision"] * num for num, metrics in eval_metrics]) / total_num)
    recall_aggregated = (sum([metrics["Recall"] * num for num, metrics in eval_metrics]) / total_num)
    f1_aggregated = (sum([metrics["F1"] * num for num, metrics in eval_metrics]) / total_num)
    auc_aggregated = (sum([metrics["AUC"] * num for num, metrics in eval_metrics]) / total_num)

    return {"auc": auc_aggregated,
            "precision": precision_aggregated,
            "recall": recall_aggregated,
            "f1_score": f1_aggregated}


def config_func(rnd: int) -> Dict[str, str]:
    # Configuration with global epochs
    return {"global_round": str(rnd)}


# Define strategy
strategy = FedXgbBagging(
    fraction_fit=(float(num_clients_per_round) / pool_size),
    min_fit_clients=num_clients_per_round,
    min_available_clients=pool_size,
    min_evaluate_clients=num_evaluate_clients,
    fraction_evaluate=1.0,
    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
    on_evaluate_config_fn=config_func,
    on_fit_config_fn=config_func)

# Start Flower server
fl.server.start_server(server_address="0.0.0.0:8080", strategy=strategy,
                       config=fl.server.ServerConfig(num_rounds=num_rounds))
