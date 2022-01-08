import sklearn.metrics as metrics
from sklearn.metrics import classification_report
from scipy.special import logit
import torch
import pickle
import numpy as np
from load_datasets import load_dataset
from preprocess import preprocessing
from utilities import train_test_split, prepare_dataset
import preprocess
from preprocess import preprocessing

torch.manual_seed(preprocess.seed)
np.random.seed(preprocess.seed)
# method I: plt
import matplotlib.pyplot as plt


def model_performance(INPUT_SHAPE, net, X_train, y_train, set_type="Training Set"):
    correct = 0
    total = 0
    prediction = torch.tensor([])
    prediction_argmax = []
    with torch.no_grad():
        output = net(X_train.view(-1, INPUT_SHAPE))
        for idx, i in enumerate(output):
            prediction = torch.cat((prediction, i))
            prediction_argmax = np.append(prediction_argmax, torch.argmax(i))
            if torch.argmax(i) == y_train[idx]:
                correct += 1
            total += 1
    print(f"Accuracy on {set_type}: ", round(correct / total, 3))
    print(classification_report(prediction_argmax, y_train))
    return prediction_argmax, output


def AUC_util(model, X_test, y_test, INPUT_SHAPE, fprs, tprs):
    # calculate the fpr and tpr for all thresholds of the classification
    _, probs = model_performance(
        INPUT_SHAPE, model, X_test, y_test, set_type="Test Set"
    )
    probs = torch.exp(probs[:, 1])
    fpr, tpr, threshold = metrics.roc_curve(y_test, probs)
    fprs = fprs.append(fpr)
    tprs = tprs.append(tpr)
    return fprs, tprs


def AUC_plot(fprs, tprs, LAYER, DATASET_NAME):
    plt.figure(figsize=(15, 7))
    plt.title(f"{DATASET_NAME.capitalize()}", fontsize=24)
    for i in range(len(fprs)):
        roc_auc = metrics.auc(fprs[i], tprs[i])
        plt.plot(fprs[i], tprs[i], label=f"AUC_{LAYER[i]} = %0.2f" % roc_auc)

    plt.legend(loc="lower right", fontsize=20)
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("$FPR_{Utility}$", fontsize=22)
    plt.ylabel("$TPR_{Utility}$", fontsize=22)
    plt.savefig(f"figures/AUC/AUC_UTILITY_NEW_{DATASET_NAME}.svg")


if __name__ == "__main__":
    models_dic = {
        "5-Layers": "Net",
        "4-Layers": "Net_Net1",
        "3-Layers": "Net_Net1_Net2",
        "2-Layers": "Net_Net1_Net2_Net3",
    }
    df, dataset_name, drop_columns, categorical_cols = load_dataset()
    df_numeric = preprocessing(df, categorical_cols)
    DATASET_NAME = dataset_name
    df_train, df_test = train_test_split(df_numeric, split=0.80)
    drop_columns_not_needed = ["y"]

    X_train, y_train, X_test, y_test = prepare_dataset(
        df_train, df_test, drop_columns_not_needed
    )
    INPUT_SHAPE = X_train.shape[1]
    fprs = []
    tprs = []
    for key in models_dic:
        LAYER = key
        MODEL_NAME = models_dic[key]
        with open(f"models/{DATASET_NAME}_{MODEL_NAME}", "rb") as f:
            model = pickle.load(f)
        AUC_util(model, X_test, y_test, INPUT_SHAPE, fprs, tprs)
    AUC_plot(fprs, tprs, LAYER, DATASET_NAME)

