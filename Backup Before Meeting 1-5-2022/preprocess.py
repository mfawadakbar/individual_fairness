"""
This File Preprocesses the data
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


seed = 1997

np.random.seed(seed)


def handle_non_numerical_data(df, categorical_cols):
    columns = categorical_cols
    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1

            df[column] = list(map(convert_to_int, df[column]))
    return df


def normalize(df_numeric):
    df_numeric = df_numeric - df_numeric.min()
    df_numeric = df_numeric / df_numeric.max()
    df_numeric = df_numeric.dropna(axis=1)
    return df_numeric


def convert_to_categorical(df, categorical_cols):
    encoder = OneHotEncoder(sparse=False)
    missing_cols = list(set(categorical_cols) - set(df.columns.values))
    categorical_cols = list(set(categorical_cols) - set(missing_cols))
    print(f"Missing Cols: {missing_cols} \n Categorical Cols: {categorical_cols}")
    # Encode Categorical Data
    df_encoded = pd.DataFrame(encoder.fit_transform(df[categorical_cols]))
    df_encoded.columns = encoder.get_feature_names(categorical_cols)

    # Replace Categotical Data with Encoded Data
    df = df.drop(categorical_cols, axis=1)
    df = pd.concat([df_encoded, df], axis=1)

    # Encode target value
    # df['y'] = df['y'].apply(lambda x: 1 if x == 'yes' else 0)

    print("Shape of dataframe:", df.shape)
    # df.head()
    return df


def preprocessing(df, categorical_cols):
    df_numeric = handle_non_numerical_data(df, categorical_cols)
    # df_numeric=df_numeric.replace(-1, 0) replace -1 with 0
    df_numeric[df_numeric < 0] = 0  # replace all values less than 0 with zero
    # df_numeric[df_numeric["marital"]
    df_numeric = normalize(df_numeric)
    df_numeric = convert_to_categorical(df_numeric, categorical_cols)
    df_numeric = df_numeric.dropna()

    return df_numeric

