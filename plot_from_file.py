import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import preprocess
from preprocess import preprocessing

np.random.seed(preprocess.seed)

ALPHAS = [i / 1000 for i in range(1, 101, 10)]
EPSILONS = [0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90]
DATSET_NAME = "bank"
MODEL = "Net"
roc_all = np.loadtxt(f"dataframes/ROC/ROC_all_{DATSET_NAME}_{MODEL}.txt")
roc_all = roc_all.reshape([110, 101, 2])
e = 0
AUC = []
roc_all = roc_all[e * 10 : (1 + e) * 10]

plt.figure(figsize=(15, 7))
for i in range(len(roc_all)):
    plt.plot(roc_all[i][:, 0], roc_all[i][:, 1], label=ALPHAS[i])
    AUC.append(metrics.auc(roc_all[i][:, 0], roc_all[i][:, 1]))
plt.title("ROC Curve", fontsize=24)
plt.xlabel("False Positive Rate", fontsize=22)
plt.ylabel("True Positive Rate", fontsize=22)
plt.xlim(0.00, 1.00)
plt.ylim(0.00, 1.00)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.savefig(f"figures/ROC/ROC_FROM_FILE_{DATSET_NAME}_{ALPHAS[-1]}_{EPSILONS[e]}.svg")
np.savetxt(
    f"dataframes/AUC/AUC_all_{DATSET_NAME}.txt", np.array(AUC),
)
