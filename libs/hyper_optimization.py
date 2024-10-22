# Imports
from catboost import CatBoostClassifier
from collections import Counter
from lightgbm import LGBMClassifier
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from xgboost import XGBClassifier
import optuna
import pandas as pd

# Default SEED
RANDOM_STATE = 42


def process_class_weight(params, trials_df, class_weight_name="class_weight"):
    # 'class_weight' as a dictionary from the class_weight_ values for each class
    params[class_weight_name] = {int(k.split("_")[-1]): v for k, v in params.items() if k.startswith(f"{class_weight_name}_")}
    # Remove previous class_weight_ from params
    params = {k: v for k, v in params.items() if not k.startswith(f"{class_weight_name}_")}

    # Columns starting with 'params_class_weight_'
    class_weight_cols = [col for col in trials_df.columns if col.startswith(f"params_{class_weight_name}_")]
    # New column 'params_class_weight' as dictionary
    trials_df[f"params_{class_weight_name}"] = trials_df[class_weight_cols].apply(
        lambda row: {int(col.split('_')[-1]): row[col] for col in class_weight_cols}, axis=1)
    # Remove previous class_weight columns
    trials_df.drop(columns=class_weight_cols, inplace=True)
    # Sort trials_df columns
    trials_df = trials_df[sorted(trials_df.columns)]

     # Hyperparameters and Trials DataFrame after adjusting class_weight
    return params, trials_df


class HyperParamRandomForestClassifier:
    def __init__(self, X: pd.DataFrame, y: pd.Series, n_trials=50, k=5, random_state=RANDOM_STATE):
        self.X = X
        self.y = y
        self.n_trials = n_trials
        self.k = k
        self.random_state = random_state

    def objective(self, trial):  # Objective Function
        # param_dist obtained from self.params
        param_dist = {"random_state": self.random_state,
                      "n_jobs": -1,
                      "n_estimators": trial.suggest_int("n_estimators", low=10, high=300, step=10),
                      "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
                      "max_depth": trial.suggest_int("max_depth", low=2, high=15),
                      "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                      "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"])}

        model = RandomForestClassifier(**param_dist)
        scores = cross_validate(estimator=model, X=self.X, y=self.y, n_jobs=-1,
                                cv=StratifiedKFold(  # Stratified K-Fold
                                    n_splits=self.k, shuffle=True, random_state=self.random_state),
                                scoring=("precision_macro", "recall_macro", "f1_macro"))
        
        print(f"[RandomForestClassifier] Optimizing: {trial.number + 1}/{self.n_trials}...", end="", flush=True)
        print("\r", end="", flush=True)
        
        # Save each trial metric
        trial.set_user_attr("test_f1_macro", scores["test_f1_macro"].mean())
        trial.set_user_attr("test_precision_macro", scores["test_precision_macro"].mean())
        trial.set_user_attr("test_recall_macro", scores["test_recall_macro"].mean())
        
        # Mean score
        return scores["test_f1_macro"].mean()

    def run(self, verbosity=optuna.logging.CRITICAL) -> RandomForestClassifier:
        optuna.logging.set_verbosity(verbosity)  # optuna.logging
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=self.random_state))
        # sampler to obtain reproducible optimization results

        # Optimizing...
        study.optimize(self.objective, n_trials=self.n_trials)
        params = study.best_trial.params  # Best hyperparameters
        trials_df = study.trials_dataframe()  # DataFrame with all trials
        print("\n\n")
        # Random Forest with best hyperparameters + best hyperparameters separately, all trials
        return RandomForestClassifier(**params), params, trials_df


class HyperParamXGBoostClassifier:
    def __init__(self, X: pd.DataFrame, y: pd.Series, n_trials=50, k=5, random_state=RANDOM_STATE):
        self.X = X
        self.y = y
        self.n_trials = n_trials
        self.k = k
        self.random_state = random_state

    def objective(self, trial):  # Objective Function
        # param_dist obtained from self.params
        param_dist = {"verbosity": 0,
                      "objective": "multi:softprob",
                      "random_state": self.random_state,
                      "learning_rate": trial.suggest_float("learning_rate", low=0.01, high=0.1, step=0.01),
                      "n_estimators": trial.suggest_int("n_estimators", low=10, high=300, step=10),
                      "max_depth": trial.suggest_int("max_depth", low=2, high=15),
                      "reg_alpha": trial.suggest_float("reg_alpha", low=0.0, high=1.0, step=0.1),
                      "reg_lambda": trial.suggest_float("reg_lambda", low=0.0, high=1.0, step=0.1),
                      "class_weight": {k: trial.suggest_int(f"class_weight_{k}", v["low"], v["high"]) for k, v in {
                          k: {"low": 1, "high": int(round(max(Counter(self.y).values()) / v, 0))} for k, v in Counter(self.y).items()}.items()}}
        
        model = XGBClassifier(**param_dist)
        scores = cross_validate(estimator=model, X=self.X, y=self.y, n_jobs=-1,
                                cv=StratifiedKFold(  # Stratified K-Fold
                                    n_splits=self.k, shuffle=True, random_state=self.random_state),
                                scoring=("precision_macro", "recall_macro", "f1_macro"))
        
        print(f"[XGBClassifier] Optimizing: {trial.number + 1}/{self.n_trials}...", end="", flush=True)
        print("\r", end="", flush=True)
        
        # Save each trial metric
        trial.set_user_attr("test_f1_macro", scores["test_f1_macro"].mean())
        trial.set_user_attr("test_precision_macro", scores["test_precision_macro"].mean())
        trial.set_user_attr("test_recall_macro", scores["test_recall_macro"].mean())
        
        # Mean score
        return scores["test_f1_macro"].mean()

    def run(self, verbosity=optuna.logging.CRITICAL) -> XGBClassifier:
        optuna.logging.set_verbosity(verbosity)  # optuna.logging
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=self.random_state))
        # sampler to obtain reproducible optimization results
        
        # Optimizing...
        study.optimize(self.objective, n_trials=self.n_trials)
        params = study.best_trial.params  # Best hyperparameters
        trials_df = study.trials_dataframe()  # DataFrame with all trials

        # Multiclass Classifier, then consider class_weight
        params, trials_df = process_class_weight(params, trials_df)

        print("\n\n")
        # XGBoost with best hyperparameters + best hyperparameters separately, all trials
        return XGBClassifier(**params), params, trials_df


class HyperParamLightGBMClassifier:
    def __init__(self, X: pd.DataFrame, y: pd.Series, n_trials=50, k=5, random_state=RANDOM_STATE):
        self.X = X
        self.y = y
        self.n_trials = n_trials
        self.k = k
        self.random_state = random_state

    def objective(self, trial):  # Objective Function
        # param_dist obtained from self.params
        param_dist = {"verbosity": -1,
                      "objective": "multiclass",
                      "boosting_type": "gbdt",
                      "random_state": self.random_state,
                      "learning_rate": trial.suggest_float("learning_rate", low=0.01, high=0.1, step=0.01),
                      "n_estimators": trial.suggest_int("n_estimators", low=10, high=300, step=10),
                      "max_depth": trial.suggest_int("max_depth", low=2, high=15),
                      "reg_alpha": trial.suggest_float("reg_alpha", low=0.0, high=1.0, step=0.1),
                      "reg_lambda": trial.suggest_float("reg_lambda", low=0.0, high=1.0, step=0.1),
                      "class_weight": {k: trial.suggest_int(f"class_weight_{k}", v["low"], v["high"]) for k, v in {
                          k: {"low": 1, "high": int(round(max(Counter(self.y).values()) / v, 0))} for k, v in Counter(self.y).items()}.items()}}
        
        model = LGBMClassifier(**param_dist, verbose_eval=False)
        scores = cross_validate(estimator=model, X=self.X, y=self.y, n_jobs=-1,
                                cv=StratifiedKFold(  # Stratified K-Fold
                                    n_splits=self.k, shuffle=True, random_state=self.random_state),
                                    scoring=("precision_macro", "recall_macro", "f1_macro"))
        
        print(f"[LGBMClassifier] Optimizing: {trial.number + 1}/{self.n_trials}...", end="", flush=True)
        print("\r", end="", flush=True)

        # Save each trial metric
        trial.set_user_attr("test_f1_macro", scores["test_f1_macro"].mean())
        trial.set_user_attr("test_precision_macro", scores["test_precision_macro"].mean())
        trial.set_user_attr("test_recall_macro", scores["test_recall_macro"].mean())
        
        # Mean score
        return scores["test_f1_macro"].mean()

    def run(self, verbosity=optuna.logging.CRITICAL) -> LGBMClassifier:
        optuna.logging.set_verbosity(verbosity)  # optuna.logging
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=self.random_state))
        # sampler to obtain reproducible optimization results

        # Optimizing...
        study.optimize(self.objective, n_trials=self.n_trials)
        params = study.best_trial.params  # Best hyperparameters
        trials_df = study.trials_dataframe()  # DataFrame with all trials

        # Multiclass Classifier, then consider class_weight
        params, trials_df = process_class_weight(params, trials_df)

        print("\n\n")
        # LightGBM with best hyperparameters + best hyperparameters separately, all trials
        return LGBMClassifier(**params, verbosity=-1), params, trials_df


class HyperParamCatBoostClassifier:
    def __init__(self, X: pd.DataFrame, y: pd.Series, n_trials=50, k=5, random_state=RANDOM_STATE):
        self.X = X
        self.y = y
        self.n_trials = n_trials
        self.k = k
        self.random_state = random_state

    def objective(self, trial):  # Objective Function
        # param_dist obtained from self.params
        param_dist = {"loss_function": "MultiClass",
                      "random_state": self.random_state,
                      "learning_rate": trial.suggest_float("learning_rate", low=0.01, high=0.1, step=0.01),
                      "n_estimators": trial.suggest_int("n_estimators", low=10, high=300, step=10),
                      "max_depth": trial.suggest_int("max_depth", low=2, high=15),
                      "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", low=0.0, high=1.0, step=0.1),
                      "class_weights": {k: trial.suggest_int(f"class_weights_{k}", v["low"], v["high"]) for k, v in {
                          k: {"low": 1, "high": int(round(max(Counter(self.y).values()) / v, 0))} for k, v in Counter(self.y).items()}.items()}}

        model = CatBoostClassifier(**param_dist, verbose=False)
        scores = cross_validate(estimator=model, X=self.X, y=self.y, n_jobs=-1,
                                cv=StratifiedKFold(  # Stratified K-Fold
                                    n_splits=self.k, shuffle=True, random_state=self.random_state),
                                scoring=("precision_macro", "recall_macro", "f1_macro"))
        
        print(f"[CatBoostClassifier] Optimizing: {trial.number + 1}/{self.n_trials}...", end="", flush=True)
        print("\r", end="", flush=True)
        
        # Save each trial metric
        trial.set_user_attr("test_f1_macro", scores["test_f1_macro"].mean())
        trial.set_user_attr("test_precision_macro", scores["test_precision_macro"].mean())
        trial.set_user_attr("test_recall_macro", scores["test_recall_macro"].mean())
        
        # Mean score
        return scores["test_f1_macro"].mean()

    def run(self, verbosity=optuna.logging.CRITICAL) -> CatBoostClassifier:
        optuna.logging.set_verbosity(verbosity)  # optuna.logging
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=self.random_state))
        # sampler to obtain reproducible optimization results

        # Optimizing...
        study.optimize(self.objective, n_trials=self.n_trials)
        params = study.best_trial.params  # Best hyperparameters
        trials_df = study.trials_dataframe()  # DataFrame with all trials

        # Multiclass Classifier, then consider class_weights
        params, trials_df = process_class_weight(params, trials_df, "class_weights")

        print("\n\n")
        # CatBoost with best hyperparameters + best hyperparameters separately, all trials
        return CatBoostClassifier(**params), params, trials_df