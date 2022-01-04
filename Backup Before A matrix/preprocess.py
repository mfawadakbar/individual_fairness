"""
This File Preprocesses the data
"""
import numpy as np


def handle_non_numerical_data(df):
    columns = df.columns.values
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


def preprocess(df):
    df_numeric = handle_non_numerical_data(df)
    # df_numeric=df_numeric.replace(-1, 0) replace -1 with 0
    df_numeric[df_numeric < 0] = 0  # replace all values less than 0 with zero
    # df_numeric[df_numeric["marital"]
    return df_numeric

