# Imports
from eli5 import show_weights
from IPython.display import display
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import cross_validate, StratifiedKFold

import matplotlib.pyplot as plt
import numpy as np
import scikitplot as skplt

# Default SEED
RANDOM_STATE = 42


def get_proba(model, X_test):
    try:
        # Get prediction_proba from model
        return model.predict_proba(X_test)
    except AttributeError:  # Function not available
        # 'Build' predict_proba from y_pred
        y_pred = model.predict(X_test)
        return np.array([[1 if i == label else 0 for i in range(len(np.unique(y_pred)))]
                        for label in y_pred])

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    fig = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    fig.figure_.suptitle(title)
    plt.show()  # Show Confusion Matrix


def plot_roc_curve(model, X_test, y_test, title="ROC Curve", figsize=(10, 10)):
    # y_proba from model
    y_proba = get_proba(model, X_test)

    # ROC Curve
    skplt.metrics.plot_roc_curve(y_test, y_proba, title=title, figsize=figsize, cmap='Set1')
    plt.show()  # Show plot


def plot_pr_curve(model, X_test, y_test, title="Precision-Recall (PR) Curve", figsize=(10, 10)):
    # y_proba from model
    y_proba = get_proba(model, X_test)

    # PR Curve
    skplt.metrics.plot_precision_recall(y_test, y_proba, title=title, figsize=figsize, cmap='Set1')
    plt.show()  # Show plot


def train_validate_model(model, X_train, y_train, k=5, random_state=RANDOM_STATE, average="", roc_auc_scoring="roc_auc", verbose=True):
    
    # Multiclass (default setting)
    if (not average or roc_auc_scoring == "roc_auc") and y_train.nunique() > 2:
        average = "_macro"
        roc_auc_scoring = "roc_auc_ovr"

    scores = cross_validate(estimator=model, X=X_train, y=y_train, n_jobs=-1,
                            cv=StratifiedKFold(  # Stratified K-Fold
                                n_splits=k, shuffle=True, random_state=random_state),
                            scoring=(f"precision{average}", f"recall{average}", f"f1{average}", f"{roc_auc_scoring}", "neg_log_loss"))  # Validação cruzada

    results = {}  # Save training/validation results to dictionary
    results["Precision"] = f"{round(scores[f'test_precision{average}'].mean() * 100, 2)} +- {round(scores[f'test_precision{average}'].std() * 100, 2)}"
    results["Recall"] = f"{round(scores[f'test_recall{average}'].mean() * 100, 2)} +- {round(scores[f'test_recall{average}'].std() * 100, 2)}"
    results["F1"] = f"{round(scores[f'test_f1{average}'].mean() * 100, 2)} +- {round(scores[f'test_f1{average}'].std() * 100, 2)}"
    results["ROC_AUC"] = f"{round(scores[f'test_{roc_auc_scoring}'].mean() * 100, 2)} +- {round(scores[f'test_{roc_auc_scoring}'].std() * 100, 2)}"
    results["Log_Loss"] = f"{round(abs(scores['test_neg_log_loss'].mean()), 4)} +- {round(abs(scores['test_neg_log_loss'].std()), 4)}"
    
    if verbose:  # Print results
        print(f"Training and Validation: {results}\n")
    
    return results  # Training/validation results


def test_model(model, X_test, y_test, cm_show=True, threshold=0.5, average="binary", average_roc_auc="macro", multi_class_roc_auc="raise", verbose=True):
    
    # Multiclass (default setting)
    if (average == "binary" or multi_class_roc_auc == "raise") and y_test.nunique() > 2:
        average = "macro"
        average_roc_auc = "macro"
        multi_class_roc_auc = "ovr"

    # Recalculate y_pred based on threshold != 0.5
    if threshold != 0.5:  # Only valid for binary classification
        y_pred_proba = get_proba(model, X_test)[:, 1]
        y_pred = (y_pred_proba > threshold).astype(int)
    else:  # Default threshold
        y_pred = model.predict(X_test)

        if multi_class_roc_auc == "raise":  # Binary
            y_pred_proba = get_proba(model, X_test)[:, 1]
        else:  # Multiclass
            y_pred_proba = get_proba(model, X_test)

    results = {}  # Save test results to dictionary
    labels = np.sort(y_test.unique())  # Sorted classes

    # Precision
    results["Precision"] = f"{round(precision_score(y_test, y_pred, average=average) * 100, 2)}"
    # Recall
    results["Recall"] = f"{round(recall_score(y_test, y_pred, average=average) * 100, 2)}"
    # F1-Score
    results["F1"] = f"{round(f1_score(y_test, y_pred, average=average) * 100, 2)}"

    # Previous metrics for each class, if multiclassification
    if y_test.nunique() > 2:
        results["Precision"] += ", " + str({f"{label}": round(precision_score(y_test, y_pred, labels=[label], average=average) * 100, 2) for label in labels})
        results["Recall"] += ", " + str({f"{label}": round(recall_score(y_test, y_pred, labels=[label], average=average) * 100, 2) for label in labels})
        results["F1"] += ", " + str({f"{label}": round(f1_score(y_test, y_pred, labels=[label], average=average) * 100, 2) for label in labels})
    
    # ROC AUC
    results["ROC_AUC"] = f"{round(roc_auc_score(y_test, y_pred_proba, average=average_roc_auc, multi_class=multi_class_roc_auc) * 100, 2)}"
    # Logarithmic Loss
    results["Log_Loss"] = f"{round(log_loss(y_test, y_pred_proba), 4)}"
    
    if verbose:  # Print results
        print(f"Model Test: {results}\n")
        # Show confusion matrix
        plot_confusion_matrix(y_test, y_pred) if cm_show else None

    return results  # Test results


def plot_feature_importances(model, feature_names, top_n=10, perc=True):
    if not hasattr(model, "feature_importances_"):
        raise ValueError(
            "Model doesn't have the 'feature_importances_!'.")

    # Get feature importances
    feature_importances = model.feature_importances_

    if perc:  # feature_importances in percentage
        feature_importances = 100.0 * \
            (feature_importances / feature_importances.sum())

    # Feature indexes in descending order of importance
    sorted_idx = feature_importances.argsort()[::-1]

    # Features and their importance
    # By default, as top_n=10, all TOP 10 features are displayed
    top_features = [feature_names[i] for i in sorted_idx[:top_n]]
    top_importances = feature_importances[sorted_idx][:top_n]

    # Generate feature importance plot
    plt.figure(figsize=(10, 6))
    plt.barh(top_features, top_importances, align="center")

    # Add importance values to the plot
    for index, value in enumerate(top_importances):
        plt.text(value, index,
                 f"{value:.2f}{'%' if perc else ''}", ha="left", va="center")

    plt.xlabel("Importance of Features")
    plt.title("Feature Importances")
    # Invert the order of features (most important at the top)
    plt.gca().invert_yaxis()
    plt.show()

    return top_features  # TOP features


def plot_weights(model, feature_names, top_n=10):
    # Weights
    display(show_weights(model, feature_names=feature_names,
                         top=top_n))