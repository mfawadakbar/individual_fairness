import numpy as np
import torch
import preprocess
from preprocess import preprocessing

torch.manual_seed(preprocess.seed)
np.random.seed(preprocess.seed)


def true_false_positive(
    threshold_vector_sim, threshold_vector_disim, y_test_sim, y_test_disim
):
    true_positive = np.equal(threshold_vector_sim, 1) & np.equal(y_test_sim, 1)
    false_positive = np.equal(threshold_vector_sim, 1) & np.equal(y_test_sim, 0)

    true_negative = np.equal(threshold_vector_disim, 0) & np.equal(y_test_disim, 0)
    false_negative = np.equal(threshold_vector_disim, 0) & np.equal(y_test_disim, 1)

    tpr = true_positive.sum() / (true_positive.sum() + false_negative.sum())
    fpr = false_positive.sum() / (false_positive.sum() + true_negative.sum())

    return tpr, fpr


def roc_from_scratch(
    probabilities_sim, probabilities_disim, y_test_sim, y_test_disim, partitions=100
):
    roc = np.array([])
    for i in range(partitions + 1):
        threshold_vector_sim = np.greater_equal(
            probabilities_sim, i / partitions
        ).astype(int)
        threshold_vector_disim = np.greater_equal(
            probabilities_disim, i / partitions
        ).astype(int)
        tpr, fpr = true_false_positive(
            threshold_vector_sim, threshold_vector_disim, y_test_sim, y_test_disim
        )
        roc = np.append(roc, [fpr, tpr])
    return roc.reshape(-1, 2)


def prob_vectors(
    net, df_numeric, indices, Xp_indices, Xn_indices, drop_columns_not_needed
):
    anchors = Xp_indices[indices[0][:]]
    positives = Xn_indices[indices[1][:]]

    anchors_X_df = df_numeric.drop(drop_columns_not_needed, axis=1).loc[anchors]
    anchors_y_df = df_numeric.loc[anchors].y

    sim_X_df = df_numeric.loc[positives].drop(drop_columns_not_needed, axis=1)
    sim_y_df = df_numeric.loc[positives].y

    anchors_X = anchors_X_df.values

    sim_X = sim_X_df.values

    prob_vector_anchor = net(
        torch.FloatTensor(anchors_X.reshape(-1, anchors_X.shape[1]))
    )[:, 1]
    prob_vector_sim = net(torch.Tensor(sim_X.reshape(-1, sim_X.shape[1])))[:, 1]

    prob_diff = np.abs(
        prob_vector_anchor.detach().numpy() - prob_vector_sim.detach().numpy()
    )

    actual_prob_diff = np.abs(anchors_y_df.values - sim_y_df.values)

    return prob_diff, actual_prob_diff


def consistency(net, df_numeric, A, Xp_indices, Xn_indices, drop_columns_not_needed):
    indices = np.where(A == 1)
    prob_diff_sim, actual_prob_diff_sim = prob_vectors(
        net, df_numeric, indices, Xp_indices, Xn_indices, drop_columns_not_needed
    )
    indices = np.where(A == 0)
    prob_diff_disim, actual_prob_diff_disim = prob_vectors(
        net, df_numeric, indices, Xp_indices, Xn_indices, drop_columns_not_needed
    )
    ROC = roc_from_scratch(
        prob_diff_sim,
        prob_diff_disim,
        actual_prob_diff_sim,
        actual_prob_diff_disim,
        partitions=100,
    )
    return ROC
