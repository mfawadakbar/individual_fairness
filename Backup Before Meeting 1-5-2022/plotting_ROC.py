import numpy as np
import matplotlib.pyplot as plt
import preprocess
from sklearn import metrics

np.random.seed(preprocess.seed)


def plot_roc(ROC_ALL, DATSET_NAME, ROC_SHAPE, ALPHAS):
    auc = []
    # roc_all = np.loadtxt(f"dataframes/ROC_all_{DATSET_NAME}.txt")
    # roc_all = roc_all.reshape(ROC_SHAPE) #(alphas, number of threshold in ROC+1 i.e. 101, number of axis i.e 2)
    roc_all = ROC_ALL
    plt.figure(figsize=(15, 7))

    for i in range(len(roc_all)):
        plt.plot(roc_all[i][:, 0], roc_all[i][:, 1], label=ALPHAS[i])
        auc.append(metrics.auc(roc_all[i][:, 0], roc_all[i][:, 1]))
    plt.title("ROC Curve", fontsize=20)
    plt.xlabel("False Positive Rate", fontsize=16)
    plt.ylabel("True Positive Rate", fontsize=16)
    plt.xlim(0.00, 1.00)
    plt.ylim(0.00, 1.00)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.savefig(f"figures/ROC_Consistency_Curve_ALL_{DATSET_NAME}_{ALPHAS[-1]}.png")
    np.savetxt(
        f"dataframes/AUC_all_{DATSET_NAME}.txt", np.array(auc),
    )
