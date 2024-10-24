# Imports
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from xgboost import XGBClassifier
import optuna
import pandas as pd

# Default SEED
RANDOM_STATE = 42


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
                      "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None])}

        model = RandomForestClassifier(**param_dist)
        scores = cross_validate(estimator=model, X=self.X, y=self.y, n_jobs=-1,
                                cv=StratifiedKFold(  # Stratified K-Fold
                                    n_splits=self.k, shuffle=True, random_state=self.random_state),
                                scoring=("recall_macro", "f1_macro"))
        
        print(f"[RandomForestClassifier] Optimizing: {trial.number + 1}/{self.n_trials}...", end="", flush=True)
        print("\r", end="", flush=True)
        
        # Mean score
        return scores["test_f1_macro"].mean()

    def run(self, verbosity=optuna.logging.CRITICAL) -> RandomForestClassifier:
        optuna.logging.set_verbosity(verbosity)  # optuna.logging
        study = optuna.create_study(direction="maximize")
        # sampler to obtain reproducible optimization results

        # Optimizing...
        study.optimize(self.objective, n_trials=self.n_trials, n_jobs=-1)
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
                      "reg_lambda": trial.suggest_float("reg_lambda", low=0.0, high=1.0, step=0.1)}
        
        model = XGBClassifier(**param_dist)
        scores = cross_validate(estimator=model, X=self.X, y=self.y, n_jobs=-1,
                                cv=StratifiedKFold(  # Stratified K-Fold
                                    n_splits=self.k, shuffle=True, random_state=self.random_state),
                                scoring=("recall_macro", "f1_macro"))
        
        print(f"[XGBClassifier] Optimizing: {trial.number + 1}/{self.n_trials}...", end="", flush=True)
        print("\r", end="", flush=True)
        
        # Mean score
        return scores["test_f1_macro"].mean()

    def run(self, verbosity=optuna.logging.CRITICAL) -> XGBClassifier:
        optuna.logging.set_verbosity(verbosity)  # optuna.logging
        study = optuna.create_study(direction="maximize")
        # sampler to obtain reproducible optimization results
        
        # Optimizing...
        study.optimize(self.objective, n_trials=self.n_trials, n_jobs=-1)
        params = study.best_trial.params  # Best hyperparameters
        trials_df = study.trials_dataframe()  # DataFrame with all trials
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
                      "reg_lambda": trial.suggest_float("reg_lambda", low=0.0, high=1.0, step=0.1)}
        
        model = LGBMClassifier(**param_dist, verbose_eval=False)
        scores = cross_validate(estimator=model, X=self.X, y=self.y, n_jobs=-1,
                                cv=StratifiedKFold(  # Stratified K-Fold
                                    n_splits=self.k, shuffle=True, random_state=self.random_state),
                                scoring=("recall_macro", "f1_macro"))
        
        print(f"[LGBMClassifier] Optimizing: {trial.number + 1}/{self.n_trials}...", end="", flush=True)
        print("\r", end="", flush=True)
        
        # Mean score
        return scores["test_f1_macro"].mean()

    def run(self, verbosity=optuna.logging.CRITICAL) -> LGBMClassifier:
        optuna.logging.set_verbosity(verbosity)  # optuna.logging
        study = optuna.create_study(direction="maximize")
        # sampler to obtain reproducible optimization results

        # Optimizing...
        study.optimize(self.objective, n_trials=self.n_trials, n_jobs=-1)
        params = study.best_trial.params  # Best hyperparameters
        trials_df = study.trials_dataframe()  # DataFrame with all trials
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
                      "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", low=0.0, high=1.0, step=0.1)}

        model = CatBoostClassifier(**param_dist, logging_level="Silent")
        scores = cross_validate(estimator=model, X=self.X, y=self.y, n_jobs=-1,
                                cv=StratifiedKFold(  # Stratified K-Fold
                                    n_splits=self.k, shuffle=True, random_state=self.random_state),
                                scoring=("recall_macro", "f1_macro"))
        
        print(f"[CatBoostClassifier] Optimizing: {trial.number + 1}/{self.n_trials}...", end="", flush=True)
        print("\r", end="", flush=True)
        
        # Mean score
        return scores["test_f1_macro"].mean()

    def run(self, verbosity=optuna.logging.CRITICAL) -> CatBoostClassifier:
        optuna.logging.set_verbosity(verbosity)  # optuna.logging
        study = optuna.create_study(direction="maximize")
        # sampler to obtain reproducible optimization results

        # Optimizing...
        study.optimize(self.objective, n_trials=self.n_trials, n_jobs=-1)
        params = study.best_trial.params  # Best hyperparameters
        trials_df = study.trials_dataframe()  # DataFrame with all trials
        print("\n\n")
        # CatBoost with best hyperparameters + best hyperparameters separately, all trials
        return CatBoostClassifier(**params, logging_level="Silent"), params, trials_df