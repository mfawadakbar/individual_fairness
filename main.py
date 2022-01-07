import numpy as np
import pandas as pd
import gc


from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from scipy.spatial import distance


import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import pickle

from ROC_fair import consistency
from utilities import (
    print_sensitive_attr,
    train_test_split,
    prepare_dataset,
    data_loaders,
    distance_raw,
    fast_psm,
)

from neural_networks import Net, Net1, Net2, Net3, train_model, model_performance
from load_datasets import load_dataset
from plotting_ROC import plot_roc
import preprocess
from preprocess import preprocessing

torch.manual_seed(preprocess.seed)
np.random.seed(preprocess.seed)


if __name__ == "__main__":
    alphas = [i / 1000 for i in range(1, 101, 10)]
    epsilons = [0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90]
    n = 1000
    n0 = 5
    n1 = 5
    learn_rate = 0.0001
    epocs = 50
    OUTPUT_SIZE = 2
    ROC_all = (
        []
    )  # np.array([6, 101, 2], dtype=np.float16)  # np.empty(shape=1, dtype=np.float16)

    df, dataset_name, drop_columns, categorical_cols = load_dataset()
    print_sensitive_attr(df, drop_columns)
    df_numeric = preprocessing(df, categorical_cols)

    df_train, df_test = train_test_split(df_numeric, split=0.80)
    # df_train = sampling(df_train)

    drop_columns_not_needed = ["y"]

    prop_diff, Xp, Xn, Xp_indices, Xn_indices = distance_raw(
        df_test, drop_columns, drop_columns_not_needed, norm_metric=np.linalg.norm,
    )

    X_train, y_train, X_test, y_test = prepare_dataset(
        df_train, df_test, drop_columns_not_needed
    )
    trainloader, testloader = data_loaders(X_train, y_train, X_test, y_test)

    trainset = trainloader
    testset = testloader

    INPUT_SHAPE = X_train.shape[1]
    print(f"Input Shape is: {INPUT_SHAPE}")
    models = [Net, Net1, Net2, Net3]

    for model in models:
        ROC_all = []
        net = model(INPUT_SHAPE, OUTPUT_SIZE)
        print(f"Working on {dataset_name}_{net.__class__.__name__}")
        dataset_name = str(f"{dataset_name}_{net.__class__.__name__}")
        train_model(
            net,
            trainset,
            learn_rate,
            loss_function=nn.CrossEntropyLoss(),
            optimizer=optim.Adam,
            epocs=epocs,
        )
        train_output = model_performance(
            INPUT_SHAPE, net, X_train, y_train, "Training Set"
        )
        test_output = model_performance(INPUT_SHAPE, net, X_test, y_test, "Test Set")
        file = open(f"dataframes/Performance/Performance_{dataset_name}.txt", "a+")
        file.writelines(train_output)
        file.writelines(test_output)
        file.close()

        # save trained model
        with open(f"models/{dataset_name}", "wb") as files:
            pickle.dump(net, files)
        # load saved model
        with open(f"models/{dataset_name}", "rb") as f:
            net = pickle.load(f)

        for epsilon in epsilons:
            for alpha in alphas:
                print(
                    f"WORKING ON DATASET {dataset_name} for alpha {alpha} for epsilon {epsilon}"
                )
                A = fast_psm(prop_diff, alpha, epsilon, n=5)
                ROC = consistency(
                    net, df_test, A, Xp_indices, Xn_indices, drop_columns_not_needed
                )
                ROC_all.append(ROC)
                gc.collect()
        file.close()
        roc_shape = np.shape(ROC_all)
        print(f"Shape of ROC_all is: {roc_shape}")
        np.savetxt(
            f"dataframes/ROC/ROC_all_{dataset_name}.txt",
            np.array(ROC_all).reshape(-1, roc_shape[1] * roc_shape[2]),
        )

        # plot_roc(ROC_all, dataset_name, roc_shape, alphas)

