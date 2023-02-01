import numpy as np
import pandas as pd



def split_train_test(df, size, time_column):
    # Build test DataFrame
    test_df = pd.DataFrame(columns=df.columns)

    for i in np.sort(pd.unique(df['class'])): # For each class
        temp_df = df[df['class'] == i] # Select only data from a class
        # Obtain a percentage of data (end of DataFrame)
        temp_df = temp_df.tail(round(len(temp_df) * size))
        # Drop data obtained from the training DataFrame
        df.drop(index=temp_df.index, inplace=True)
        # Add data in test DataFrame
        test_df = pd.concat([test_df, temp_df])

    # Sort by time column and reset index
    # df is training (and validating) DataFrame
    df = df.sort_values(by=[time_column]).reset_index(drop=True)
    test_df = test_df.sort_values(by=[time_column]).reset_index(drop=True)
    # Return df and test_df excluding time column
    return df.drop(columns=[time_column]), test_df.drop(columns=[time_column])



def split_x_y(df, columns_to_drop):
    # Columns to drop
    df = df.drop(columns=columns_to_drop)   
    # DataFrame X (features)
    X = df.loc[:, df.columns != 'class']
    y = df.loc[:, 'class'] # y (labels)
    # Return DataFrame X and y
    return X, y



def load_data(data_path, size=0.2):
    # Load dataset
    df = pd.read_csv(data_path)
    # Split the data into 80% for training and 20% for test (default)
    train_df, test_df = split_train_test(df, size, 'timestamp')
    # Split train_df into X and y
    X_train, y_train = split_x_y(train_df, [])
    # Split train_df into X and y
    X_test, y_test = split_x_y(test_df, [])
    # GPS spoofing and jamming as a single category
    y_test = y_test.replace(2, 1)
    # Return train and test dataset
    return X_train, y_train, X_test, y_test
