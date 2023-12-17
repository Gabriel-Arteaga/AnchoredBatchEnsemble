# -*- coding: utf-8 -*-
"""
These functions are written by Alexander Sabelström. You can find his work on github under "Sabelz".
Repo: https://github.com/Sabelz/Gaussian_Processes_Inducing_Points
"""
# Imports
import os
import sys
sys.path.append('..')
import numpy as np
import torch
import gpytorch
import math
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from gpytorch.priors import LogNormalPrior
from gpytorch.priors import GammaPrior
import time
import pandas as pd
from utils.metrics import calculate_PIC_PIW

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, init_lengthscale):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # Initialize the RBF kernel and set the lengthscale prior
        rbf_kernel = gpytorch.kernels.RBFKernel()
        #rbf_kernel.lengthscale = LogNormalPrior(0, init_lengthscale)

        self.covar_module = gpytorch.kernels.ScaleKernel(rbf_kernel)
        self.covar_module.base_kernel.lengthscale = init_lengthscale


    # GP Posterior predictive distribution
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
def train(model, xTrain, yTrain, epochs): # Train the model on training data: xTrain, yTrain
  """
    Train the Gaussian Process (GP) model on training data.

    Parameters:
    - model (gpytorch.models.ExactGP): The GP model to be trained.
    - xTrain (torch.Tensor): Input training data tensor of size (n x d), where n is the number of samples and d is the number of features.
    - yTrain (torch.Tensor): Output training data tensor of size (n).

  """
  smoke_test = ('CI' in os.environ)
  training_iter = 2 if smoke_test else epochs


  # Find optimal model hyperparameters
  model.train()
  model.likelihood.train()

  # Use the adam optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

  # "Loss" for GPs - the marginal log likelihood
  mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
  # Train without printing to ensure the training method is as fast as possible
  for i in range(training_iter):
      # Zero gradients from previous iteration
      model.likelihood.train()
      optimizer.zero_grad()
      # Output from model
      output = model(xTrain)
      # Calc loss and backprop gradients
      loss = -mll(output, yTrain)
      loss.backward()
      optimizer.step()

         
         

def kmeansPoints(x, y, N): # The dataset (X,Y) and the N amount of inducing points wished
  """
    Apply K-means clustering to obtain N inducing points from the input dataset (X, Y).

    Parameters:
    - x (torch.Tensor): Input data tensor of size (n x d), where n is the number of samples and d is the number of features.
    - y (torch.Tensor): Output data tensor of size (n), corresponding to the labels or values for each input sample.
    - N (int): Number of inducing points desired.

    Returns:
    - torch.Tensor: Tensor containing the inducing points obtained from K-means clustering, with size (N x d).
    - torch.Tensor: Corresponding y values for each inducing point, with size (N).
  """
  RS = 0  # Random state

  # Move tensor to CPU
  x_cpu = x.cpu().numpy()

  kmeans = KMeans(n_clusters=N, n_init=1, random_state=RS).fit(x_cpu)
  xInducing = kmeans.cluster_centers_

  # To get the corresponding y values for each inducing point, compute the closest data point.
  closest_indices, _ = pairwise_distances_argmin_min(xInducing, x_cpu)
  yInducing = y[closest_indices]

  # Move tensors back to CUDA
  xInducing_cuda = torch.from_numpy(xInducing).float().to(x.device)
  yInducing_cuda = yInducing.to(x.device)

  return xInducing_cuda, yInducing_cuda

def model_all_points(x_train, y_train, x_test, y_test, epochs):
    """
    Train a Gaussian Process (GP) model on the given training data and evaluate its performance on the test data.

    Parameters:
    - x_train (torch.Tensor): Input training data tensor of size (n_train x d), where n_train is the number of training samples and d is the number of features.
    - y_train (torch.Tensor): Output training data tensor of size (n_train), corresponding to the labels or values for each training sample.
    - x_test (torch.Tensor): Input test data tensor of size (n_test x d), where n_test is the number of test samples and d is the number of features.
    - y_test (torch.Tensor): Output test data tensor of size (n_test), corresponding to the labels or values for each test sample.

    Returns:
    - float: Execution time (in seconds) for training the GP model.
    - float: Root Mean Squared Error (RMSE) on the test data.
    - float: Probability Calibration Index (PICP) on the test data.
    - float: Mean Prediction Interval Width (MPIW) on the test data.
    - GPModel: Trained GP model.
    - float: Execution time (in seconds) for making predictions on the test data.
    """

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood() # Decide likelihood
    init_lengthscale = 1
    model = GPModel(x_train, y_train, likelihood, init_lengthscale = 1)
    model = model.to(device) # Move model to device
    start_time = time.time() # Start time
    train(model, x_train, y_train, epochs) # Train the model
    end_time = time.time() # End time
    execution_time_train = end_time - start_time # Calcualte execution time

    # Performance
    model.eval() # eval mode is for computing predictions through the model posterior.
    model.likelihood.eval()

    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var(): # https://arxiv.org/abs/1803.06058
      start_time = time.time() # Start time
      observed_pred =  model.likelihood(model(x_test))# gives us the posterior predictive distribution p(y* | x*, X, y) which is the probability distribution over the predicted output value
      end_time = time.time() # End time
      execution_time_posterior = end_time - start_time # Calculate execution time

      prediction_means = observed_pred.mean # Mean
      prediction_variances = observed_pred.variance # Variance

      squared_error = (prediction_means - y_test)**2 # Compute the squared error
      mean_SE = squared_error.mean() # Compute the mean squared error
      RMSE = math.sqrt(mean_SE) # Compute the square root of the mean squared error

      # PCIP and MPIW
      pic,piw, n = calculate_PIC_PIW(0,0, 0, prediction_means, prediction_variances, y_test)
      picp = pic/n
      mpiw = piw/n

      return execution_time_train, RMSE, picp, mpiw, model, execution_time_posterior

def model_inducing_half(x_train, y_train, x_test, y_test, epochs):
    """
    Train a Gaussian Process (GP) model with half of the training points as Inducing Points and evaluate its performance on the test data.

    Parameters:
    - x_train (torch.Tensor): Input training data tensor of size (n_train x d), where n_train is the number of training samples and d is the number of features.
    - y_train (torch.Tensor): Output training data tensor of size (n_train), corresponding to the labels or values for each training sample.
    - x_test (torch.Tensor): Input test data tensor of size (n_test x d), where n_test is the number of test samples and d is the number of features.
    - y_test (torch.Tensor): Output test data tensor of size (n_test), corresponding to the labels or values for each test sample.

    Returns:
    - float: Total execution time (in seconds) for K-means clustering and training the GP model with Inducing Points.
    - float: Root Mean Squared Error (RMSE) on the test data.
    - float: Probability Calibration Index (PICP) on the test data.
    - float: Mean Prediction Interval Width (MPIW) on the test data.
    - GPModel: Trained GP model with Inducing Points.
    - float: Execution time (in seconds) for making predictions on the test data.
    """
    # Model with half training points being Inducing Points
    points = int(len(x_train)/2) # how many Inducing Points
    start_time = time.time() # Start time
    inducingPointsX, inducingPointsY = kmeansPoints(x_train, y_train, points) # Calculate Incucing Points
    end_time = time.time() # End time
    execution_time_kmeans = end_time - start_time # Calcualte execution time for K-means

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood() # Decide likelihood
    init_lengthscale = 1
    model_inducing = GPModel(inducingPointsX, inducingPointsY, likelihood, init_lengthscale = 1) # Send in inducing points as the training points
    model_inducing = model_inducing.to(device) # Move model to device
    start_time = time.time() # Start time
    train(model_inducing, inducingPointsX, inducingPointsY, epochs) # Train the model
    end_time = time.time() # End time
    execution_time_train_inducing = end_time - start_time # Calcualte execution time

    # Performance
    model_inducing.eval() # eval mode is for computing predictions through the model posterior.
    model_inducing.likelihood.eval()

    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var(): # https://arxiv.org/abs/1803.06058
      start_time = time.time() # Start time
      observed_pred_inducing =  model_inducing.likelihood(model_inducing(x_test))# gives us the posterior predictive distribution p(y* | x*, X, y) which is the probability distribution over the predicted output value
      end_time = time.time() # End time
      posterior_time_inducing = end_time - start_time # Calculate execution time

      prediction_means_inducing = observed_pred_inducing.mean # Mean
      prediction_variances_inducing = observed_pred_inducing.variance # Variance
      squared_error = (prediction_means_inducing - y_test)**2 # Compute the squared error
      mean_SE = squared_error.mean() # Compute the mean squared error
      RMSE_inducing = math.sqrt(mean_SE) # Compute the square root of the mean squared error

      # PCIP and MPIW
      pic,piw, n = calculate_PIC_PIW(0,0, 0, prediction_means_inducing, prediction_variances_inducing, y_test)
      picp_inducing = pic/n
      mpiw_inducing = piw/n

      training_time = execution_time_kmeans+execution_time_train_inducing # Total training time
      return training_time, RMSE_inducing, picp_inducing, mpiw_inducing, model_inducing, posterior_time_inducing

def test_GP_inference_time(model, x_test, n_iterations):
  """
    Test the inference time of a Gaussian Process (GP) model on the given test data.

    Parameters:
    - model (gpytorch.models.ExactGP): Trained GP model.
    - x_test (torch.Tensor): Input test data tensor of size (n_test x d), where n_test is the number of test samples and d is the number of features.
    - n_iterations (int): Number of iterations for testing the inference time.

    Returns:
    - list: List of inference times (in seconds) for each iteration.
  """
  inference_times = []
  for _ in range(n_iterations):
    start_time = time.time()
    with torch.no_grad():
      # Make prediction
      output = model(x_test)
      end_time = time.time()
      inference_times.append(end_time-start_time)
  return inference_times

   