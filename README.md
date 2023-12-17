## BatchEnsemble and Anchored Regularization (PyTorch)
This repository presents a PyTorch implementation of BatchLinear as introduced in the *BatchEnsemble* paper by [Wen et al.](https://arxiv.org/abs/2002.06715).  
Additionally, it features a hybrid implementation combining BatchEnsemble with the Anchored regularization method proposed in the *Approximately Bayesian* paper by [Pearce et al.](https://proceedings.mlr.press/v108/pearce20a.html).

## Table Of Contents
 * [Ensemble Background](#Ensemble-Background)
 * [Why BatchEnsemble and Anchored Regularization?](#BatchEnsemble-and-Anchored-Regularization)
 * [Prior Strategy](#Prior-Strategy)
 * [Toy Data results](#Toy-Data)
 * [Experiments](#Experiments)
 * [Usage](#Usage)
## Background
Deep Neural Network (DNNs) are powerful predictors, however, it is not seldom that they produce wrong predictions overconfidently.
Several solutions have been proposed to counteract this behavior, quantifying the uncertainty is one important step. 
Bayesian Neural Networks (BNNs) utilizes methods such as MCMC to place probability distributions over the network weights, although effective for quantifying uncertainty it comes with a huge computational cost. Deep Ensembles were first proposed by [Lakshminarayanan et al.](https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html) as a scaleable non-Bayesian alternative to BNNs for quantifying uncertainty. Deep Ensembles utilizes multiple neural networks independent predictions and lets the variance of the predictions represent the uncertainty of the model.
<p align="center">
<img src="data/readme_pics/deep_ensemble.png" width="600"/>
</p>

## BatchEnsemble and Anchored Regularization
Although Deep Ensembles are more scaleable than their BNN counterpart, the lack of Bayesian framework has been critized, additionally a Deep Ensemble's cost for both training and testing increases linearly with the number of ensemble members.  
*BatchEnsemble* introduces a memory efficient way to add ensemble members to an ensemble, it achieves this by cleverly defining a weight matrix as the Hardmard product of one shared weight matrix shared among ensemble members and a rank-one matrix per ensemble member: 
$$\bar{W}_i = W \odot F_i, \text{where }F_i=r_is_i^T$$

## Prior Strategy

## Toy Data

## Experiments

## Usage
