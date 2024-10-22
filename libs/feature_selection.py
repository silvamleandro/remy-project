# Imports
from BorutaShap import BorutaShap
import numpy as np
import pandas as pd

# Default SEED
RANDOM_STATE = 42


def get_iv_woe(data, target_column="is_target", bins=10, verbose=False, show_woe=False):
    # Learn more: https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html
    # Empty dataframe for calculations
    new_df, woe_df = pd.DataFrame(), pd.DataFrame()
    # Extract column names
    cols = data.columns

    # WOE and IV on all independent variables
    for ivars in cols[~cols.isin([target_column])]:
        if (data[ivars].dtype.kind in "bifc") and (len(np.unique(data[ivars])) > 10):
            binned_x = pd.qcut(data[ivars], bins, duplicates="drop")
            d0 = pd.DataFrame({"x": binned_x, "y": data[target_column]})
        else:
            d0 = pd.DataFrame({"x": data[ivars], "y": data[target_column]})

        d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
        d.columns = ["Cutoff", "N", "Events"]
        d["% of Events"] = np.maximum(d["Events"], 0.5) / d["Events"].sum()
        d["Non-Events"] = d["N"] - d["Events"]
        d["% of Non-Events"] = np.maximum(d["Non-Events"],
                                          0.5) / d["Non-Events"].sum()
        d["WoE"] = np.log(d["% of Events"] / d["% of Non-Events"])
        d["IV"] = d["WoE"] * (d["% of Events"] - d["% of Non-Events"])
        d.insert(loc=0, column="Variable", value=ivars)

        if verbose:
            print("Information value of " + ivars +
                  " is " + str(round(d["IV"].sum(), 6)))

        temp = pd.DataFrame({"Variable": [ivars], "IV": [
                            d["IV"].sum()]}, columns=["Variable", "IV"])
        new_df = pd.concat([new_df, temp], axis=0)
        woe_df = pd.concat([woe_df, d], axis=0)

        # Display WoE table
        if show_woe:
            print(d)

    # Return  DataFrames new_df and woe_df
    return new_df, woe_df


def get_high_corr_features_pairwise(df, threshold=0.8, method="spearman"):
    cor_matrix = df.corr(method=method).abs()
    # Series with highly correlated features...
    return cor_matrix[cor_matrix > threshold].unstack().sort_values(ascending=False).drop_duplicates()


def select_feature_boruta(X, y, model, random_state=RANDOM_STATE, n_trials=50):
    # Feature selector configured for classification
    Feature_Selector = BorutaShap(
        model=model, importance_measure="shap", classification=True)
    # Select features with n_trials
    Feature_Selector.fit(X=X, y=y, n_trials=n_trials,
                         random_state=random_state)
    # Features Boxplot
    Feature_Selector.plot(which_features="all",
                          figsize=(16, 12), y_scale="log")
    # Accept or reject undecided features
    Feature_Selector.TentativeRoughFix()