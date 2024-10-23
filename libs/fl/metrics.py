# Imports
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score
)


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