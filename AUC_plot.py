import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import preprocess

np.random.seed(preprocess.seed)

ALPHAS = [i / 1000 for i in range(1, 101, 10)]
EPSILONS = [0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90]
DATASET_NAME = "german"
MODEL = "Net"
AUC = []
auc_all = np.loadtxt(f"dataframes/AUC/AUC_all_{DATASET_NAME}.txt")
auc_all = auc_all.reshape([11, 10])
plt.figure(figsize=(15, 7))
sns.set(font_scale=2)
sns.heatmap(
    auc_all,
    xticklabels=ALPHAS,
    yticklabels=EPSILONS,
    vmin=0,
    vmax=1,
    annot=False,
    fmt="f",
)
plt.xlabel("$Alpha$", fontsize=18)
plt.ylabel("$Epsilon$", fontsize=18)
plt.title(f"{DATASET_NAME.capitalize()} Dataset", fontsize=24)
# plt.xlim(0.001, 0.091)
# plt.ylim(0.70, 0.90)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig(f"figures/AUC/AUC_FROM_FILE_{DATASET_NAME}.svg")

