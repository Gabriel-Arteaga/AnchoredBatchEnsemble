## BatchEnsemble and Anchored Regularization (PyTorch)
This repository presents a PyTorch implementation of BatchLinear as introduced in the *BatchEnsemble* paper by [Wen et al.](https://arxiv.org/abs/2002.06715).  
Additionally, it features a hybrid implementation combining BatchEnsemble with the Anchored regularization method proposed in the *Approximately Bayesian* paper by [Pearce et al.](https://proceedings.mlr.press/v108/pearce20a.html).

## Table Of Contents
 * [Ensemble Background](#Ensemble-Background)
 * [Why BatchEnsemble and Anchored Regularization?](#BatchEnsemble-and-Anchored-Regularization)
 * [Experiments](#Experiments)
 * [Usage](#Usage)
 * [Acknowledgements](#Acknowledgements)
## Background
Deep Neural Network (DNNs) are powerful predictors, however, it is not seldom that they produce wrong predictions overconfidently.
Several solutions have been proposed to counteract this behavior, quantifying the uncertainty is one important step. 
Bayesian Neural Networks (BNNs) utilizes methods such as MCMC to place probability distributions over the network weights, although effective for quantifying uncertainty it comes with a huge computational cost. Deep Ensembles were first proposed by [Lakshminarayanan et al.](https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html) as a scaleable non-Bayesian alternative to BNNs for quantifying uncertainty. Deep Ensembles utilizes multiple neural networks independent predictions and lets the variance of the predictions represent the uncertainty of the model.
<p align="center">
<img src="data/readme_pics/deep_ensemble.png" width="600"/>
</p>

## BatchEnsemble and Anchored Regularization
Although Deep Ensembles are more scalable than their BNN counterpart, the lack of Bayesian framework has been criticized, additionally a Deep Ensemble's cost for both training and testing increases linearly with the number of ensemble members.

*Anchored Regularization* provides an approximation of Bayesian inference. In our proposed method, we employ the following loss function, incorporating anchored regularization for each ensemble member $j$:
<p align="center">
$Loss_j=\frac{1}{N}GNLLL_j+\frac{1}{N}||\pmb{\tau}^{1/2}\cdot(\pmb{\theta}_j-\pmb{\theta}$<sub>$anc, j$</sub>) $||^2_2$
<p>
The right term in the loss function represents the anchored regularization term, where $\pmb{\theta}$<sub>$anc, j$</sub> is drawn from a Gaussian distribution:
<p align="center">
    $\pmb{\theta}$<sub>$anc, j$</sub> $\sim \mathcal{N}(\pmb{\mu}$<sub>$prior$</sub>$, \pmb{\Sigma}$<sub>$prior$</sub>)
<p>
The left term corresponds to the Gaussian Negative Log-Likelihood Loss (GNLLL), enabling our network to predict both mean and variance terms. This enabled an effective disentanglement of aleatoric and epistemic uncertainties, combining them into a predictive uncertainty
<p float="left">
  <img src="/data/readme_pics/Ensemble_Wiggle.png" width="49%" height="400" alt="Naive ensemble method" />
  <img src="/data/readme_pics/anchored_wiggle.png" width="49%" height="400" alt="Our proposed method" /> 
</p>
<p>
  <em> &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Naive Ensemble Method &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Our proposed method.</em>
</p>



We leveraged *BatchEnsemble* as a memory efficient way to add ensemble members to an ensemble. This is achieved by defining a weight matrix as the Hardmard product of one shared weight matrix shared among ensemble members and a rank-one matrix per ensemble member: 
$$\bar{W}_i = W \odot F_i, \text{where }F_i=r_is_i^T$$
We assessed our proposed method, incorporating BatchEnsemble, against a Naive ensemble that requires sequential training. The difference in training time becomes substantial as the number of ensemble members increases, as illustrated in the following plot.
<p align="center">
  <img src="/data/readme_pics/training_time.png" width="49%" height="400" alt="Naive ensemble method" />
</p>
<p align="center">
<em> Training time for different ensemble sizes for our proposed ensemble and the Naive ensemble </em>
</p>


## Experiments
In our project, we conducted benchmark experiments using five different models on two UCI regression datasets. Three of these models were neural network ensembles: our proposed method, **'Anchored Batch'**, the BatchEnsemble model by Wen et al. named **'Batch'**, and a **'Naive'** ensemble where each ensemble was trained sequentially. It's important to note that all ensemble models shared the same architecture hyperparameters and loss functions. The only distinction between 'Batch' and 'Naive' was the scalability improvements, while 'AnchoredBatch' additionally employed anchored regularization. In addition to the neural network ensembles, we benchmarked two types of Gaussian Processes—one with all data points (**'GP'**) and another utilizing an inducing point strategy containing half the number of points (**'GP Induce'**).  
Below you case see our results on the [Power](https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant) dataset:

| **Metric** | **Anchored Batch** | **Batch** | **Naive** | **GP** | **GP Induce** |
|:----------:|:------------------:|:---------:|:---------:|:------:|:-------------:|
| RMSE ↑     | **4.24**           | 4.27      | 4.86      | 5.11   | 5.80          |
| PICP ↑     | **0.96**               | 0.99  | 1.00      | 0.76   | 0.69          |
| MPIW ↓     | **17.05**          | 22.09     | 48.77     | 12.20  | 11.91         |
| Train Time (s) ↓ | 456.06      | 333.64    | 1755.97 | 87.55  | **52.92**         |
| Inference Time (s) ↓ | **0.02**    | **0.02**      | 0.04      | 0.31   | 0.10      |  

Note that we employed $2\sigma$ prediction intervals, equivalent to a 95% prediction interval. Our objective in generating high-quality prediction intervals was to minimize the Mean Prediction Interval Width (MPIW) while ensuring a Prediction Interval Coverage Probability (PICP) of 95%. With this in mind we can see that **'Anchored Batch'** produced better results in all categories, except in training time. However, note that **'Anchored Batch'** outperformed the **'Naive'** model substantially in terms of train time.  
The second dataset we utilized was the [Concrete](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) dataset:
| Metric | Anchored Batch | Batch | Naive | GP | GP Induce |
|--------|----------------|-------|-------|----|-----------|
| RMSE ↑ | 6.98           | 6.85  | **5.59**  | 7.45 | 7.92      |
| PICP ↑ | 0.92           | 0.91  | **0.98**  | 0.73 | 0.60      |
| MPIW ↓ | 24.26          | 23.39 | **26.40** | 15.17 | 14.90     |
| Train Time (s) ↓ | 75.59 | 50.99 | 496.42 | 24.49 | **5.01**   |
| Inference Time (s) ↓ | 0.003 | **0.002** | 0.147 | 0.008 | 0.004 |

This result was somewhat disappointing as the **'Naive'** model outperformed the other models in all metrics except for train time and inference time. We attribute this outcome, in part, to the fact that the experiments were only run once; with more iterations, the average might yield a different result. It's crucial to emphasize that all ensembles used the same loss function and architecture, with the naive model differing only in terms of the regularization term from the **'Anchored Batch'**. Nevertheless, further investigation into the behavior of this result will be deferred to future work.  
These experiments were produced by running the `experiments.py` file.




## Usage
Run the `usage_example.ipynb` notebook to get an overview of using the BatchLinear layer and creating a BatchEnsemble in PyTorch. Additionally, explore the initialization and training process for the proposed method, AnchoredBatchEnsemble.
## Acknowledgements
First and foremost, I would like to thank my supervisor [Nicolas Pielawski](https://github.com/npielawski) , for inviting me to participate in this project, introducing me to the intricacies of uncertainty in machine learning, and providing valuable guidance throughout the project.  
Additionally, I would like to express my appreciation to my project group member, [Alexander Sabelström](https://github.com/Sabelz). He dedicated his efforts to the Gaussian Processes aspect of the experiments, and you can find his contributions in the following [repository](https://github.com/Sabelz/Gaussian_Processes_Inducing_Points).
