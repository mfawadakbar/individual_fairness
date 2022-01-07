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
    plt.title("ROC Curve", fontsize=24)
    plt.xlabel("False Positive Rate", fontsize=22)
    plt.ylabel("True Positive Rate", fontsize=22)
    plt.xlim(0.00, 1.00)
    plt.ylim(0.00, 1.00)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18)
    plt.savefig(f"figures/ROC/ROC_Consistency_Curve_ALL_{DATSET_NAME}_{ALPHAS[-1]}.svg")
    np.savetxt(
        f"dataframes/AUC/AUC_all_{DATSET_NAME}.txt", np.array(auc),
    )


"""
import numpy as np
import matplotlib.pyplot as plt

alphas = [i / 1000 for i in range(1, 101, 10)]

roc_all = np.loadtxt(f"dataframes/ROC_all_german_Net.txt")
roc_all = roc_all.reshape([110,101,2])
plt.figure(figsize=(15, 7))
for i in range(len(roc_all)[:10]): plt.plot(roc_all[i][:, 0], roc_all[i][:, 1], label=alphas[i])

plt.show()

roc_all = roc_all.reshape([10,11,101,2])

"""
