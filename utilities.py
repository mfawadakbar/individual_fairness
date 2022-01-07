import numpy as np
import random
from preprocess import preprocessing
import preprocess
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import pandas as pd
from load_datasets import find_patterns

random.seed(preprocess.seed)
np.random.seed(preprocess.seed)
torch.manual_seed(preprocess.seed)


def print_sensitive_attr(df_numeric, drop_columns):
    columns = df_numeric.columns.to_list()
    column = drop_columns[0]
    print("The columns are: ", columns)
    print("")
    print(
        "Unique values in the column ",
        "\033[1m",
        column,
        "\033[0m",
        " is",
        np.sort(df_numeric[column].unique()),
    )
    print("")
    print(
        "Lowest value in the column ",
        "\033[1m",
        column,
        "\033[0m",
        " is ",
        min(df_numeric[column]),
    )
    print("")
    print(
        "Highest value in the column ",
        "\033[1m",
        column,
        "\033[0m",
        "is",
        max(df_numeric[column]),
    )
    print("")
    print(
        "Assigned categories to the column ",
        "\033[1m",
        column,
        "\033[0m",
        " is ",
        np.sort(df_numeric[column].unique()),
    )

    np.sum(df_numeric.isna())


def balanced_psm(
    df_train,
    prop_diff,
    Xp,
    Xn,
    Xp_indices,
    Xn_indices,
    alpha=0.01,
    epsilon=0.8,
    n0=5,
    n1=5,
):
    sim, dif = {}, {}
    y_temp = df_train.y
    for i in tqdm(range(len(Xp))):
        t0, t1 = 1, 1
        q0, q1 = 1, 1
        for j in range(len(Xn)):
            adjusted_xp = Xp_indices[i]
            adjusted_xn = Xn_indices[j]
            distance = prop_diff[i][j]
            if distance < alpha:
                if not adjusted_xp in sim:
                    sim[adjusted_xp] = []
                elif y_temp[adjusted_xn] == 0 and t0 <= n0:
                    sim[adjusted_xp].append(adjusted_xn)
                    t0 += 1
                elif y_temp[adjusted_xn] == 1 and t1 <= n1:
                    sim[adjusted_xp].append(adjusted_xn)
                    t1 += 1
            elif distance > epsilon:
                if not adjusted_xp in dif:
                    dif[adjusted_xp] = []
                elif y_temp[adjusted_xn] == 0 and q0 <= n0:
                    # print("q",q)
                    dif[adjusted_xp].append(adjusted_xn)
                    q0 += 1
                elif y_temp[adjusted_xn] == 1 and q1 <= n1:
                    # print("q",q)
                    dif[adjusted_xp].append(adjusted_xn)
                    q1 += 1
    return sim, dif


def unbalanced_psm(
    df_train, prop_diff, Xp, Xn, Xp_indices, Xn_indices, n=1000, alpha=0.01, epsilon=0.8
):
    sim, dif = {}, {}
    for i in tqdm(range(len(Xp))):
        t = 1
        q = 1
        for j in range(len(Xn)):
            adjusted_xp = Xp_indices[i]
            adjusted_xn = Xn_indices[j]

            distance = prop_diff[i][j]
            if distance < alpha:
                if not adjusted_xp in sim:
                    sim[adjusted_xp] = []
                elif t <= n:
                    sim[adjusted_xp].append(adjusted_xn)
                    t += 1
                    # print("t",t)

            elif distance > epsilon:
                if not adjusted_xp in dif:
                    dif[adjusted_xp] = []
                elif q <= n:
                    # print("q",q)
                    dif[adjusted_xp].append(adjusted_xn)
                    q += 1
    return sim, dif


def make_pairs(A, df_train, Xp_indices, Xn_indices):
    indices = np.where(A == 1)
    anchors = Xp_indices[indices[0][:]]
    positives = Xn_indices[indices[1][:]]
    pairs_sim = pd.DataFrame([anchors, positives], columns=["Anchor", "Positive"])
    pairs_sim.sort_values(by=["Anchor"])
    temp_value = (
        np.sum(df_train.loc[pairs_sim["Positive"]].y)
        / df_train.loc[pairs_sim["Positive"]].shape[0]
    )
    print(f"Percentage of y=1 in pairs_sim: {temp_value}")  # Fix error here

    indices = np.where(A == 0)
    anchors = Xp_indices[indices[0][:]]
    negetives = Xn_indices[indices[1][:]]
    pairs_df = pd.DataFrame([anchors, negetives], columns=["Anchor", "Negetive"])
    pairs_df.sort_values(by=["Negetive"])
    temp_value = (
        np.sum(df_train.loc[pairs_df["Negetive"]].y)
        / df_train.loc[pairs_df["Negetive"]].shape[0]
    )
    print(f"Percentage of y=1 in pairs_df: {temp_value}")
    return pairs_sim, pairs_df


def export(pairs_sim, pairs_df, dataset_name, alpha, epsilon, n, n0, n1):
    fig, ax = plt.subplots(figsize=(100, 15))
    if pairs_df.shape[0] != 0:
        pairs_df["Negetive"].value_counts().plot(kind="bar")
        plt.title("Negetive Pairs with their Indexes", fontsize=20)
        plt.xlabel("Indexes", fontsize=16)
        plt.ylabel("Frequency of each Index", fontsize=16)
        plt.savefig(
            f"figures/Neg_Indexes_{dataset_name}_{alpha}_{epsilon}_{n}_{n0}_{n1}.png"
        )

    fig, ax = plt.subplots(figsize=(100, 15))
    if pairs_sim.shape[0] != 0:
        pairs_sim["Positive"].value_counts().plot(kind="bar")
        plt.title("Positive Pairs with their Indexes", fontsize=20)
        plt.xlabel("Indexes", fontsize=16)
        plt.ylabel("Frequency of each Index", fontsize=16)
        plt.savefig(
            f"figures/Pos_Indexes_{dataset_name}_{alpha}_{epsilon}_{n}_{n0}_{n1}.png"
        )

    pairs_df.to_csv(
        f"dataframes/pairs_df_{dataset_name}_{alpha}_{epsilon}_{n}_{n0}_{n1}.csv"
    )
    pairs_sim.to_csv(
        f"dataframes/pairs_sim_{dataset_name}_{alpha}_{epsilon}_{n}_{n0}_{n1}.csv"
    )


def make_triplets(pairs_sim, pairs_df, dataset_name, alpha, epsilon, n, n0, n1):
    Ai = list(pairs_sim["Anchor"].values)
    Aj = list(pairs_sim["Positive"].values)
    Az = []
    for x in tqdm(Ai):
        Az.append(pairs_df[pairs_df["Anchor"] == x]["Negetive"].values)

    Az = list(np.concatenate(Az).astype(np.int32))

    print("Ai shape: ", np.shape(Ai))
    print("Aj shape: ", np.shape(Aj))
    print("Az shape: ", np.shape(Az))

    # Make a dataframe of dissimilar pairs
    df_matching_pairs = pd.DataFrame(
        list(zip(Ai, Aj, Az)), columns=["Anchor", "Positive", "Negetive"]
    )
    df_matching_pairs.to_csv(
        f"dataframes/df_matching_triplets_{dataset_name}_{alpha}_{epsilon}_{n}_{n0}_{n1}.csv"
    )
    return df_matching_pairs


def export_viz_pairs(df_train, pairs_sim, dataset_name, alpha, epsilon, n, n0, n1):
    case00, case01, case10, case11 = 0, 0, 0, 0
    for i in tqdm(range(len(pairs_sim))):
        a, p = pairs_sim.iloc[i]
        if df_train.loc[p].y == 0 and df_train.loc[a].y == 0:
            case00 += 1
        elif df_train.loc[p].y == 0 and df_train.loc[a].y == 1:
            case01 += 1
        elif df_train.loc[p].y == 1 and df_train.loc[a].y == 0:
            case10 += 1
        elif df_train.loc[p].y == 1 and df_train.loc[a].y == 1:
            case11 += 1

    fig, ax = plt.subplots(figsize=(16, 9))
    plt.bar(["case00", "case01", "case10", "case11"], [case00, case01, case10, case11])
    plt.title("Pairs Cases Frequency", fontsize=20)
    plt.xlabel("Pairs Cases", fontsize=16)
    plt.ylabel("Frequency of Each Case", fontsize=16)
    plt.savefig(
        f"figures/pairs_cases_{dataset_name}_{alpha}_{epsilon}_{n}_{n0}_{n1}.png"
    )


def export_viz_triplets(
    df_train, df_matching_pairs, dataset_name, alpha, epsilon, n, n0, n1
):
    case000, case001, case010, case011 = 0, 0, 0, 0
    case100, case101, case110, case111 = 0, 0, 0, 0
    for i in tqdm(range(len(df_matching_pairs))):
        a, p, n = df_matching_pairs.iloc[i]
        if df_train.loc[a].y == 0 and df_train.loc[p].y == 0 and df_train.loc[n].y == 0:
            case000 += 1
        elif (
            df_train.loc[a].y == 0 and df_train.loc[p].y == 0 and df_train.loc[n].y == 1
        ):
            case001 += 1
        elif (
            df_train.loc[a].y == 0 and df_train.loc[p].y == 1 and df_train.loc[n].y == 0
        ):
            case010 += 1
        elif (
            df_train.loc[a].y == 0 and df_train.loc[p].y == 1 and df_train.loc[n].y == 1
        ):
            case011 += 1
        elif (
            df_train.loc[a].y == 1 and df_train.loc[p].y == 0 and df_train.loc[n].y == 0
        ):
            case100 += 1
        elif (
            df_train.loc[a].y == 1 and df_train.loc[p].y == 0 and df_train.loc[n].y == 1
        ):
            case101 += 1
        elif (
            df_train.loc[a].y == 1 and df_train.loc[p].y == 1 and df_train.loc[n].y == 0
        ):
            case110 += 1
        elif (
            df_train.loc[a].y == 1 and df_train.loc[p].y == 1 and df_train.loc[n].y == 1
        ):
            case111 += 1

    fig, ax = plt.subplots(figsize=(16, 9))
    plt.bar(
        [
            "case000",
            "case001",
            "case010",
            "case011",
            "case100",
            "case101",
            "case110",
            "case111",
        ],
        [case000, case001, case010, case011, case100, case101, case110, case111],
    )
    plt.title("Triplet Cases Frequency", fontsize=20)
    plt.xlabel("Triplet Cases", fontsize=16)
    plt.ylabel("Frequency of Each Case", fontsize=16)
    plt.savefig(
        f"figures/triplets_cases_{dataset_name}_{alpha}_{epsilon}_{n}_{n0}_{n1}.png"
    )


def blob_label(y, label, loc):  # assign labels
    target = np.copy(y)
    for l in loc:
        target[y == l] = label
    return target


def train_test_split(df_numeric, split=0.70):

    indexes_0 = np.array(df_numeric[df_numeric.y == 0].index)
    random.Random(1997).shuffle(indexes_0)
    indexes_1 = np.array(df_numeric[df_numeric.y == 1].index)
    random.Random(1997).shuffle(indexes_1)

    train_indexes_0 = indexes_0[: int((len(indexes_0) * split))]
    train_indexes_1 = indexes_1[: int((len(indexes_1) * split))]
    test_indexes_0 = indexes_0[int((len(indexes_0) * split)) :]
    test_indexes_1 = indexes_1[int((len(indexes_1) * split)) :]

    df_train_class_0 = df_numeric[df_numeric.y == 0].loc[train_indexes_0]
    df_train_class_1 = df_numeric[df_numeric.y == 1].loc[train_indexes_1]
    df_test_class_0 = df_numeric[df_numeric.y == 0].loc[test_indexes_0]
    df_test_class_1 = df_numeric[df_numeric.y == 1].loc[test_indexes_1]
    print("df_train_class_0.shape", df_train_class_0.shape)
    print("df_train_class_1.shape", df_train_class_1.shape)
    print("df_test_class_0.shape", df_test_class_0.shape)
    print("df_test_class_1.shape", df_test_class_1.shape)

    df_train = pd.concat([df_train_class_0, df_train_class_1])
    df_test = pd.concat([df_test_class_0, df_test_class_1])

    print("Train Set y Count: \n", df_train.y.value_counts())
    print("Test Set y Count: \n", df_test.y.value_counts())

    return df_train, df_test


def prepare_dataset(df_train, df_test, drop_columns_not_needed):
    X_train = df_train.drop(drop_columns_not_needed, axis=1).values
    y_train = df_train["y"].values
    X_test = df_test.drop(drop_columns_not_needed, axis=1).values
    y_test = df_test["y"].values

    # CREATE RANDOM DATA POINTS
    X_train = torch.FloatTensor(X_train)
    y_train = torch.Tensor(y_train).type(torch.LongTensor)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.Tensor(y_test).type(torch.LongTensor)

    print(
        "Number of people who were rejected (Train): ", np.argwhere(y_train == 0).shape
    )
    print(
        "Number of people who were accepted: (Train) ", np.argwhere(y_train == 1).shape
    )
    print("Number of people who were rejected: (Test) ", np.argwhere(y_test == 0).shape)
    print("Number of people who were accepted: (Test) ", np.argwhere(y_test == 1).shape)

    return X_train, y_train, X_test, y_test


def data_loaders(X_train, y_train, X_test, y_test):
    train_data = []
    for i in range(len(X_train)):
        train_data.append([X_train[i], y_train[i]])
    trainloader = torch.utils.data.DataLoader(train_data, shuffle=False, batch_size=64)

    test_data = []
    for i in range(len(X_test)):
        test_data.append([X_test[i], y_test[i]])
    testloader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=64)

    return trainloader, testloader


def sampling(df_train):
    print("Before sampling y is: \n", df_train.y.value_counts())
    count_class_1, count_class_0 = df_train.y.value_counts()
    max_count = np.max([count_class_1, count_class_0])
    df_class_0 = df_train[df_train.y == 0]
    df_class_1 = df_train[df_train.y == 1].sample(
        max_count, replace=True, random_state=1997
    )
    print(df_class_0.shape)
    print(df_class_1.shape)
    df_train = pd.concat([df_class_0, df_class_1], ignore_index=True, sort=False)
    print("After sampling y is: \n", df_train.y.value_counts())
    return df_train


def distance_raw(
    df_train, drop_columns, drop_columns_not_needed, norm_metric=np.linalg.norm
):
    pd.options.mode.chained_assignment = None  # This supresses the copy warning, which says use view instead of copy to avoid data misuse. Use iloc to avoid this error https://towardsdatascience.com/how-to-suppress-settingwithcopywarning-in-pandas-c0c759bd0f10
    sensitive_attr = []
    sensitive_attr = find_patterns(df_train, drop_columns, sensitive_attr)
    print(
        "Inside distance_raw - Sensitive Attribute value count: ",
        df_train[sensitive_attr[0]].value_counts(),
    )
    Xp = df_train[df_train[sensitive_attr[0]] == 1].drop(
        drop_columns_not_needed, axis=1
    )
    Xn = df_train[df_train[sensitive_attr[0]] == 0].drop(
        drop_columns_not_needed, axis=1
    )
    Xp_indices = Xp.index
    Xn_indices = Xn.index
    Xn = norm_metric(Xn.values, axis=1)
    Xp = norm_metric(Xp.values, axis=1)

    print("Shape of Xp is: ", Xp.shape)
    print("Shape of Xn is: ", Xn.shape)

    prop_diff = np.empty(shape=(len(Xp), len(Xn)), dtype=np.float16)
    for i in range(len(Xp)):
        prop_diff[i] = np.abs((Xp[i] - Xn.reshape(-1)))
        # prop_diff[i] = distance.cosine(Xp[i], Xn.reshape(-1))

    prop_diff = (prop_diff - np.min(prop_diff)) / np.max(prop_diff)

    return prop_diff, Xp, Xn, Xp_indices, Xn_indices


def fast_psm(prop_diff, alpha=0.01, epsilon=0.8, n=5):

    A = np.empty(shape=list(np.shape(prop_diff)), dtype=np.float16)
    prop_diff.sort(axis=1)
    sim_indices = np.where(prop_diff < alpha)

    dis_indices = np.where(prop_diff > epsilon)

    A[sim_indices[0][:], sim_indices[1][:]] = 1
    A[dis_indices[0][:], dis_indices[1][:]] = 0

    return A

