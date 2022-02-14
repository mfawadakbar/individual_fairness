# Individual Fairness Evaluation Metric for Probablistic Classifiers
This repository is linked to a manuscript submitted to 5th AAAI/ACM Conference on AI, Ethics, and Society.

Paper Title: "PCIndFair: A New Framework to Assess the Individual Fairness of Probabilistic Classifiers"

It consists of three major components. First, training a probabilistic model on four datasets and then evaluating its fairness using a novel framework based on an oracle similarity matrix and 𝐴𝑈𝐶𝑓𝑎𝑖𝑟. The 𝐴𝑈𝐶𝑓𝑎𝑖𝑟 is computed from the 𝑅𝑂𝐶𝑓𝑎𝑖𝑟 curve of the False Positive and True Positive rates of fair and unfair predictions. Next, we explain each component in detail,

## Abstract:
<i>
Fairness in Machine Learning (ML) has become a global concern due to the predominance of ML in automated decision-making systems. Besides group fairness, individual fairness, which ensures that similar individuals are treated similarly, has received limited attention due to its associated challenges. One major challenge is the availability of a proper metric to evaluate individual fairness, especially for probabilistic classifiers. In this study, we propose a framework PCIndFair to precisely assess the individual fairness of probabilistic classifiers. We assume an oracle matrix as ground truth for similar and dissimilar pairs. PCIndFair quantifies the degree of fairness using Receiver Operating Characteristic (ROC) curve, i.e. 𝑅𝑂𝐶𝑓𝑎𝑖𝑟 and Area Under the Curve (AUC) measure, i.e. 𝐴𝑈𝐶𝑓𝑎𝑖𝑟. Our framework considers probability distribution rather than the
final classification outcome and is not dependent on a cut-off threshold like the notable consistency. measure of individual fairness. Experimental evaluations on four standard datasets reflect the theoretical benefits of the framework. Among four datasets, four different Artificial Neural Networks model architectures and different values of similarity threshold (𝛼) and dissimilarity threshold (𝜀) are also evaluated to dig deeper into the framework. The study is helpful for fairness researchers and practitioners to assess the fairness of their models and select proper parameters for their datasets. The complete code of the framework will be publicly available upon publication.
</i>


**Copyright (C) 2022 DSA Lab, Computer Science, Utah State University**
