# Import libraries
import sys
sys.path.append('..')
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
from typing import Union, Optional, Tuple
import torch
from torch import nn
import pandas as pd
import copy
import time
from torch.utils.tensorboard import SummaryWriter
from utils.models import BatchEnsemble, AnchoredBatchEnsemble
from utils.classes import EarlyStopping
from utils.metrics import calculate_PIC_PIW
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prediction function orginally created by wjmaddox\drbayes
def plot_predictive(data: np.ndarray, trajectories: np.ndarray, xs: np.ndarray, mu: np.ndarray = None,
                    sigma: np.ndarray = None, title: str = None) -> None:
    """
    Plots predictive trajectories with uncertainty bands.

    Parameters:
    - data (np.ndarray): Data points for scatter plot.
    - trajectories (np.ndarray): Predictive trajectories.
    - xs (np.ndarray): X-axis values for the trajectories.
    - mu (np.ndarray, optional): Mean values for the predictive trajectories. If not provided, it's calculated from `trajectories`.
    - sigma (np.ndarray, optional): Standard deviation values for the predictive trajectories. If not provided, it's calculated from `trajectories`.
    - title (str, optional): Title for the plot.

    Returns:
    None

    Example:
    ```python
    data = np.array([[1, 2], [2, 3], [3, 4]])
    trajectories = np.random.randn(10, 100)
    xs = np.linspace(0, 10, 100)
    plot_predictive(data, trajectories, xs, title='Predictive Plot')
    ```
    """
    sns.set_style('darkgrid')
    palette = sns.color_palette('colorblind')
    
    blue = sns.color_palette()[0]
    red = sns.color_palette()[3]

    plt.figure(figsize=(9., 7.))
    
    plt.plot(data[:, 0], data[:, 1], "o", color=red, alpha=0.7, markeredgewidth=1., markeredgecolor="k")
    if mu is None:
        mu = np.mean(trajectories, axis=0)
    if sigma is None:
        sigma = np.std(trajectories, axis=0)

    plt.plot(xs, mu, "-", lw=2., color=blue)
    plt.plot(xs, mu-3 * sigma, "-", lw=0.75, color=blue)
    plt.plot(xs, mu+3 * sigma, "-", lw=0.75, color=blue)
    np.random.shuffle(trajectories)
    for traj in trajectories[:10]:
        plt.plot(xs, traj, "-", alpha=.5, color=blue, lw=1.)
        
    plt.fill_between(xs, mu-3*sigma, mu+3*sigma, alpha=0.35, color=blue)

    plt.xlim([np.min(xs), np.max(xs)])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if title:
        plt.title(title, fontsize=16)

def plot_predictive_2(data: np.ndarray, means: np.ndarray, variances: np.ndarray, ensemble_size: int, xs: np.ndarray, title: str = None, pred_interval: float=99.7) -> None:
    """
    Plots predictive mean, aleatoric, epistemic and combined uncertainty bands given arrays of means and variances.

    Parameters:
    - data (np.ndarray): Data points for scatter plot.
    - means (np.ndarray): Predicted means for each ensemble member.
    - variances (np.ndarray): Predicted variances for each ensemble member.
    - ensemble_size (int): Number of ensemble members.
    - xs (np.ndarray): X-axis values for the trajectories.
    - title (str, optional): Title for the plot.
    - pred_interval (float): Prediction interval in interval should be a float in range (0,100). Default (99.7)

    Returns:
    None
    """
    z = compute_z_score(pred_interval)
    sns.set_style('darkgrid')
    palette = sns.color_palette('colorblind')
    
    blue = sns.color_palette()[0]
    red = sns.color_palette()[3]
    black = 'black'

    plt.figure(figsize=(9., 7.))
    
    plt.plot(data[:, 0], data[:, 1], "o", color=red, alpha=0.7, markeredgewidth=1., markeredgecolor="k")
    
    mu = np.mean(means, axis=0)
    aleatoric_uncertainty = np.sum(variances, axis=0) / ensemble_size
    epistemic_uncertainty = (1/ensemble_size*np.sum(means**2,axis=0))-(1/ensemble_size*np.sum(means,axis=0))**2
    sigma = np.sqrt(aleatoric_uncertainty)

    plt.plot(xs, mu, "-", lw=1., color=black)
    plt.plot(xs, mu-z * sigma, "-", lw=0.75, color=blue)
    plt.plot(xs, mu+z * sigma, "-", lw=0.75, color=blue)
    plt.fill_between(xs, mu-z*sigma, mu+z*sigma, alpha=0.5, color=blue)

    plt.plot(xs, mu, "-", lw=2., color=black)
    plt.plot(xs, mu-z * np.sqrt(epistemic_uncertainty), "--", lw=0.75, color=blue)
    plt.plot(xs, mu+z * np.sqrt(epistemic_uncertainty), "--", lw=0.75, color=blue)
    plt.fill_between(xs, mu-z*np.sqrt(epistemic_uncertainty), mu+z*np.sqrt(epistemic_uncertainty), alpha=0.7, color=blue, hatch='+++')

    combined_uncertainty = np.sqrt(aleatoric_uncertainty) + np.sqrt(epistemic_uncertainty)
    plt.plot(xs, mu, "-", lw=1., color=black)
    plt.plot(xs, mu-z * combined_uncertainty, "-", lw=0.75, color=red)
    plt.plot(xs, mu+z * combined_uncertainty, "-", lw=0.75, color=red)
    plt.fill_between(xs, mu-z*combined_uncertainty, mu+z*combined_uncertainty, alpha=0.10, color=red)

    plt.xlim([np.min(xs), np.max(xs)])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Create custom artists for the legend
    aleatoric_patch = mpatches.Patch(color=blue, alpha=0.5, label='Aleatoric Uncertainty')
    epistemic_patch = mpatches.Patch(color=blue, alpha=0.7, hatch='+++', label='Epistemic Uncertainty')
    combined_patch = mpatches.Patch(color=red, alpha=0.10, label='Combined Uncertainty')
    mean_line = mlines.Line2D([], [], color=black, label='Combined mean') 

    # Add these custom artists to the legend
    plt.legend(handles=[aleatoric_patch, epistemic_patch, combined_patch, mean_line], fontsize=12)

    if title:
        plt.title(title, fontsize=16)


 
def train_models(
    model: torch.nn.Module,
    ensemble_type: str,
    ensemble_size: int,
    data_noise: float,
    epochs: int,
    print_frequency: int,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    test_loader: Optional[torch.utils.data.DataLoader],
    test: bool=False,
    early_stopping: bool=False,
    patience: int=10,
    min_delta: float=0.0,
    learning_rate: Optional[float]=0.001,
    weight_decay: Optional[float]=None,
    device: torch.device = device) -> Union[None, float]:
    """
    Train neural network models using various ensemble strategies.

    Args:
        ensemble_type (str): The ensemble strategy ('batch', 'anchored_batch', 'naive').
        ensemble_size (int): Number of ensemble members (relevant for 'batch' and 'anchored_batch' ensembles).
        data_noise (float): Magnitude of noise to be added to the data for 'anchored_batch' ensemble.
        epochs (int): Number of training epochs.
        print_frequency (int): Frequency of printing training progress.
        loss_fn (torch.nn.Module): Loss function for optimization.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        train_loader (torch.utils.data.DataLoader): DataLoader providing training data.
        test_loader (Optional[torch.utils.data.DataLoader]): DataLoader providing test data (default is None).
        test (bool): True or False, Shall we measure test 
        early_stopping (bool): Stop early according to patience?
        patience (int): Number of turns with no improvement until early stopping is triggered (default is 10). 
        min_delta (float): Minimum change in validation loss to be considered an improvement (default is 0.0)
        learning_rate (float): Learning rate for optimization.
        device (torch.device, optional): Device for training (default is device).

    Returns:
         Union[None, float]: If test is False or test_loader is None, returns None. 
                           If test is True and test_loader is provided, returns the average test loss.
    """
    # Still need to implement EarlyStopping for normal model architectures.
    if early_stopping:
                ES = EarlyStopping(patience=patience,min_delta=min_delta)
    if ensemble_type == 'batch' or ensemble_type == 'anchored_batch':
        if weight_decay == None:
            optimizer = optimizer(model.parameters(),lr=learning_rate)
        else:
            optimizer = optimizer(model.parameters(),lr=learning_rate, weight_decay=weight_decay)
        model.to(device)
        for epoch in range(epochs):
            # Training
            train_loss = 0
            # Add a loop to loop through the training batches
            for batch, (X, y) in enumerate(train_loader):
                model.train()
                # 1. Perform forward pass
                y_pred = model(X.expand(ensemble_size,1)) # Make prediction

                # 2. Calculate loss per batch
                loss = loss_fn(y_pred, y.expand(ensemble_size,1)) # Calculate loss with MSE

                train_loss += loss.item() # Accumalate loss

                if ensemble_type == 'anchored_batch':
                    batch_size = X.shape[0]
                    # Calculate regular
                    reg_term = 0
                    for layer in model.layer_stack.modules():
                    # If layer is not activation function, then it has weights
                        if hasattr(layer, 'weight'):
                            reg_term += layer.get_reg_term(batch_size, data_noise)
                    loss += reg_term
                # 3. Optimizer zero grad
                optimizer.zero_grad() # Set the optimizer's gradients to zero

                # 4. Loss backward
                loss.backward()

                #5. Optimizer step
                optimizer.step()

            if epoch%print_frequency == 0:
                print(f"Epoch: {epoch}\n-------")
                print("Loss:", train_loss / (batch + 1))  # Calculate and print average loss per batch
            # Evaluation on test data
        
        if test and test_loader is not None:
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for batch, (X_test, y_test) in enumerate(test_loader):
                    y_pred_test = model(X_test.expand(ensemble_size, 1))
                    test_loss += loss_fn(y_pred_test, y_test.expand(ensemble_size, 1)).item()

            average_test_loss = test_loss / (batch + 1)
            print(f"\nEvaluation on Test Data\n------------------------")
            print("Average Test Loss:", average_test_loss)
            
            return average_test_loss
    
    if ensemble_type == 'naive':
        # If naive there are several models we need to sequentially train
        for Model in model:
            # We're optimizing each model's parameters
            Model_Optimizer = optimizer(Model.parameters(), lr= learning_rate)
            model.to(device)

            # Conduct training
            for epoch in range(epochs):
                # Training
                train_loss = 0
                # Add a loop to loop through the training batches
                for batch, (X, y) in enumerate(train_loader):
                    model.train()
                    # 1. Perform forward pass
                    y_pred = model(X) # Make prediction

                    # 2. Calculate loss per batch
                    loss = loss_fn(y_pred, y) # Calculate loss with MSE

                    train_loss += loss.item() # Accumalate loss

                    # 3. Optimizer zero grad
                    optimizer.zero_grad() # Set the optimizer's gradients to zero

                    # 4. Loss backward
                    loss.backward()

                    #5. Optimizer step
                    optimizer.step()
                if epoch%print_frequency == 0:
                    print(f"Epoch: {epoch}\n-------")
                    print("Loss:", train_loss / (batch + 1))  # Calculate and print average loss per batch

        # Evaluation on test data for each model
        if test and test_loader is not None:
            test_loss = 0
            for Model in model:
                Model.eval()
                Model_test_loss = 0
                with torch.no_grad():
                    for batch, (X_test, y_test) in enumerate(test_loader):
                        y_pred_test = Model(X_test)
                        Model_test_loss += loss_fn(y_pred_test, y_test).item()
                    average_test_loss += Model_test_loss/ (batch+1)
            average_test_loss /= ensemble_size
            print(f"\nEvaluation on Test Data\n------------------------")
            print("Average Test Loss:", average_test_loss)
                
            return average_test_loss



def calculate_uncertainties(means: torch.Tensor, variances: torch.Tensor, ensemble_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate aleatoric, epistemic, and total uncertainties.

    Args:
    - means (torch.Tensor): Tensor of shape (data_size,1) containing mean predictions.
    - variances (torch.Tensor): Tensor of shape (data_size,1) containing model variances.
    - ensemble_size (int): Number of ensemble members.

    Returns:
    - aleatoric_uncertainty (torch.Tensor): Tensor of shape (data_size,) representing aleatoric uncertainty.
    - epistemic_uncertainty (torch.Tensor): Tensor of shape (data_size,) representing epistemic uncertainty.
    - total_uncertainty (torch.Tensor): Tensor of shape (data_size,) representing total uncertainty.
    """
    # Compute the different uncertainties
    aleatoric_uncertainty = torch.sum(variances, axis=1) / ensemble_size
    epistemic_uncertainty = (1/ensemble_size*torch.sum(means**2,axis=1))-(1/ensemble_size*torch.sum(means,axis=1))**2
    total_uncertainty = aleatoric_uncertainty+epistemic_uncertainty

    return aleatoric_uncertainty, epistemic_uncertainty, total_uncertainty

def train_models_w_mean_var(
    model: torch.nn.Module,
    ensemble_type: str,
    ensemble_size: int,
    data_noise: float,
    epochs: int,
    print_frequency: int,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    test_loader: Optional[torch.utils.data.DataLoader],
    test: bool=False,
    RMSE: bool=False,
    ENCE: bool=False,
    early_stopping: bool=False,
    calibration_metrics: bool=False,
    patience: int=10,
    min_delta: float=0.0,
    learning_rate: Optional[float]=0.001,
    weight_decay: Optional[float]=None,
    writer: Optional[torch.utils.tensorboard.SummaryWriter]=None,
    device: torch.device = device) -> Optional[Union[float, Tuple[float, float]]]:
    """
    Args:
        model (torch.nn.Module): The neural network model to be trained.
        ensemble_type (str): The ensemble strategy ('batch', 'anchored_batch', 'naive').
        ensemble_size (int): Number of ensemble members (relevant for 'batch' and 'anchored_batch' ensembles).
        data_noise (float): Magnitude of noise to be added to the data for 'anchored_batch' ensemble.
        epochs (int): Number of training epochs.
        print_frequency (int): Frequency of printing training progress.
        loss_fn (torch.nn.Module): Loss function for optimization.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        train_loader (torch.utils.data.DataLoader): DataLoader providing training data.
        test_loader (Optional[torch.utils.data.DataLoader]): DataLoader providing test data (default is None).
        test (bool): True or False, Shall we measure on test data,
        early_stopping (bool): Do we want to use early_stopping to reduce overfitting? 
        calibration_metrics (bool): If true returns MPIW and PICP metrics
        patience (int): Number of turns with no improvement until early stopping is triggered (default is 10). 
        min_delta (float): Minimum change in validation loss to be considered an improvement (default is 0.0)
        RMSE (bool): If True will calculate RMSE
        learning_rate (float): Learning rate for optimization.
        device (torch.device, optional): Device for training (default is device).

    Returns:
        Optional[Union[float, Tuple[float, float]]]: If `test` is False or `test_loader` is None, returns None.
        If `test` is True and `test_loader` is provided, returns the average test loss.
        If `RMSE` is True, additionally returns the Root Mean Squared Error (RMSE).
    """
    if early_stopping:
        ES = EarlyStopping(patience=patience,min_delta=min_delta)
    # If RMSE initiate MSE loss function
    if RMSE:
        MSE_loss_fn = nn.MSELoss(reduction='none')
    
    # Initiate a variable to track best model performance
    best_loss = None

    # Record the start time for training
    start_time_training = time.time()

    # Initiate variable which stores test timer
    test_time = 0
    if ensemble_type == 'batch' or ensemble_type == 'anchored_batch':
        optimizer = optimizer(model.parameters(), lr=learning_rate)
        model.to(device)

        for epoch in range(epochs):
            # Training
            train_loss = 0
            # Add a loop to loop through the training batches
            for batch, (X, y) in enumerate(train_loader):
                model.train()
                batch_size = X.shape[0]
                # 1. Perform forward pass
                mean, var = model(X.unsqueeze(1).expand(-1, ensemble_size, -1))  # Make prediction

                # 2. Calculate loss per batch
                loss = loss_fn(mean, y.unsqueeze(1).expand(-1, ensemble_size, -1), var) 

                train_loss += loss.item()  # Accumulate loss

                if ensemble_type == 'anchored_batch':
                    # Calculate regularization term
                    reg_term = 0
                    for layer in model.layer_stack.modules():
                        # If layer is not an activation function, then it has weights
                        if hasattr(layer, 'weight'):
                            reg_term += layer.get_reg_term(batch_size, data_noise)
                    loss += reg_term

                # 3. Optimizer zero grad
                optimizer.zero_grad()
                # 4. Loss backward
                loss.backward()
                # 5. Optimizer step
                optimizer.step()
            

            test_time_start = time.time()
            with torch.no_grad():
                test_loss = 0
                for batch, (X_test, y_test) in enumerate(test_loader):
                    mean_test, var_test = model(X_test.unsqueeze(1).expand(-1, ensemble_size, -1))
                    test_loss += loss_fn(mean_test, y_test.unsqueeze(1).expand(-1, ensemble_size, -1), var_test).item()
                average_test_loss = test_loss / (batch + 1)
                if best_loss is None or average_test_loss < best_loss:
                    # Update model's best performance
                    best_loss= average_test_loss
                    # Save the model's weights w/ best performance
                    best_model = copy.deepcopy(model.state_dict())
            # Save information to tensorboard
            if writer != None:
                writer.add_scalar('Loss/train', train_loss/(batch+1), epoch)
                writer.add_scalar('Loss/test', average_test_loss, epoch)
        
            # Check if we should stop early
            if early_stopping:
                done = ES(model,average_test_loss, epoch)
            if epoch % print_frequency == 0:
                print(f"Epoch: {epoch}\n-------")
                print("Train Loss:", train_loss / (batch + 1))  # Calculate and print average loss per batch
                if early_stopping:
                    print('Test loss:',average_test_loss, ES.status)
            if early_stopping:
                if done:
                    # Set epoch to the epoch which achieves best test loss
                    epoch = ES.epoch
                    # Early stop
                    break
            # Add the time taken to test the model performance against test data
            test_time_end = time.time()
            test_time += test_time_end-test_time_start
        # Calculate training time
        end_time_training = time.time()
        training_time = end_time_training-start_time_training-test_time

        if test and test_loader is not None:
            # Load the weights w/ the best performance
            model.load_state_dict(best_model)
            model.eval()
            test_loss = 0
            ensemble_test_loss = 0
            if calibration_metrics:
                # Initialize variables for calculating PICP and MPIW
                n = 0
                pic = 0.0
                piw = 0.0
            # If True initiate MSE Loss
            if RMSE:
                rmse_loss = 0
                ensemble_rmse_loss = 0
            with torch.no_grad():
                for batch, (X_test, y_test) in enumerate(test_loader):
                    mean_test, var_test = model(X_test.unsqueeze(1).expand(-1, ensemble_size, -1))
                    test_loss += loss_fn(mean_test, y_test.unsqueeze(1).expand(-1, ensemble_size, -1), var_test).item()
                    
                    # Calculate combined ensemble's loss
                    aleatoric_uncertainty, epistemic_uncertainty, combined_uncertainty = calculate_uncertainties(mean_test,var_test,ensemble_size)
                    # Made changes to calculate uncertainties, should return tensors of size (batch_size, 1) now instead of (batch_size)
                    ensemble_mean = torch.mean(mean_test, dim=1)
                    # Might be something fishy with combined_uncertainty dimensions with changes made look up
                    ensemble_test_loss += loss_fn(ensemble_mean, y_test, combined_uncertainty)

                    # If True Calculate the RMSE
                    if RMSE:
                        MSE = MSE_loss_fn(mean_test, y_test.unsqueeze(1).expand(-1, ensemble_size, -1))
                        ensemble_MSE = MSE_loss_fn(ensemble_mean, y_test)
                        rmse_loss += torch.sqrt(MSE.mean()).item()
                        ensemble_rmse_loss += torch.sqrt(ensemble_MSE.mean()).item()
                    if calibration_metrics:
                        pic,piw, n = calculate_PIC_PIW(pic,piw, n, ensemble_mean, combined_uncertainty, y_test)

            # Compute the average over the batches
            average_test_loss = test_loss / (batch + 1)
            average_ensemble_test_loss = ensemble_test_loss / (batch + 1)
            print(f"\nEvaluation on Test Data\n------------------------")
            print("Average Test Loss:", average_test_loss)
            if RMSE:
                average_rmse_loss = rmse_loss / (batch + 1)
                average_ensemble_rmse_loss = ensemble_rmse_loss / (batch + 1)
                if calibration_metrics:
                    # Calculate Prediction Interval Coverage Probabiblity (PICP)
                    PICP = pic/n
                    # Calculate Mean Prediction Interval Width (MPIW)
                    MPIW = piw/n
                    return average_test_loss, average_rmse_loss, epoch, average_ensemble_test_loss, average_ensemble_rmse_loss, training_time, PICP, MPIW
                return average_test_loss, average_rmse_loss, epoch, average_ensemble_test_loss, average_ensemble_rmse_loss, training_time
            else:
                if calibration_metrics:
                    return average_test_loss, epoch, average_ensemble_test_loss, training_time, PICP, MPIW 
                return average_test_loss, epoch, average_ensemble_test_loss, training_time


    # Need to implement early stop and ENCE for naive ensemble type
    elif ensemble_type == 'naive':
        # If naive, there are several models we need to sequentially train
        for Model in model:
            # We're optimizing each model's parameters
            Model_Optimizer = torch.optim.AdamW(Model.parameters(), lr=learning_rate)
            model.to(device)

            # Conduct training
            for epoch in range(epochs):
                # Training
                train_loss = 0
                # Add a loop to loop through the training batches
                for batch, (X, y) in enumerate(test_loader):
                    model.train()
                    # 1. Perform forward pass
                    mean, var = model(X)  # Make prediction

                    # 2. Calculate loss per batch
                    loss = loss_fn(mean, y, var)  # Calculate loss with GaussianNLLLoss

                    train_loss += loss.item()  # Accumulate loss

                    # 3. Optimizer zero grad
                    Model_Optimizer.zero_grad()  # Set the optimizer's gradients to zero

                    # 4. Loss backward
                    loss.backward()

                    # 5. Optimizer step
                    Model_Optimizer.step()

                if epoch % print_frequency == 0:
                    print(f"Epoch: {epoch}\n-------")
                    print("Loss:", train_loss / (batch + 1))  # Calculate and print average loss per batch

        # Evaluation on test data for each model
        if test and test_loader is not None:
            test_loss = 0
            for Model in model:
                Model.eval()
                Model_test_loss = 0
                if RMSE:
                    Model_rmse_loss = 0
                with torch.no_grad():
                    for batch, (X_test, y_test) in enumerate(test_loader):
                        mean_test, var_test = Model(X_test.expand(ensemble_size, 1))
                        Model_test_loss += loss_fn(mean_test, y_test.unsqueeze(1).expand(batch_size, ensemble_size, 1), var_test).item()
                        # If True calculate RMSE
                        if RMSE:
                            Model_rmse_loss += torch.sqrt(MSE_loss_fn(mean_test, y_test.unsqueeze(1).expand(batch_size, ensemble_size, 1))).item()
                    average_test_loss += Model_test_loss/ (batch+1)
                    if RMSE:
                        average_rmse_loss += Model_rmse_loss / (batch+1)
            average_test_loss /= ensemble_size
            print(f"\nEvaluation on Test Data\n------------------------")
            print("Average Test Loss:", average_test_loss)
            if RMSE:
                average_rmse_loss /= ensemble_size
                return average_test_loss, average_rmse_loss
            else:
                return average_test_loss



def train_and_save_results(
        model_name: str,
        hidden_layers_options: list,
        hidden_units_options: list,
        input_shape: int,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        ensemble_size: int,
        epochs: int,
        csv_file: str,
        print_frequency: Optional[int] = 500,
        weight_decay_options: Optional[list] = None,
        data_noise_options: Optional[list] = None,
        test: bool = True,
        early_stopping: bool =True,
        RMSE: bool = True,
        ENCE: bool = True,
        learning_rate: Optional[float] = 0.001,
        tensorboard_directory: Optional[str] = 'runs',
        device: torch.device = device
) -> None:
    """
    Train a model with specified configurations and save the results to a CSV file.

    Parameters:
    - model_name (str): Name of the model, e.g., 'naive', 'batch', or 'anchored_batch'.
    - hidden_layers_options (list): List of integers representing the range of hidden layers to be considered.
    - hidden_units_options (list): List of integers representing the hidden units to be considered.
    - input_shape (int): Number of features in dataset
    - loss_fn (torch.nn.Module): PyTorch loss function.
    - optimizer (torch.optim.Optimizer): PyTorch optimizer.
    - train_loader (torch.utils.data.DataLoader): PyTorch DataLoader for training data.
    - test_loader (torch.utils.data.DataLoader): PyTorch DataLoader for test data.
    - ensemble_size (int): Number of models in the ensemble.
    - epochs (int): Number of training epochs.
    - csv_file (str): Name of the CSV file to save the results.
    - print_frequency (int, optional): Frequency of printing training information. Default is 500.
    - weight_decay_options (list, optional): List of floats representing the weight decay values to be considered. Default is None.
    - data_noise_options (list, optional): List of floats representing the data noise levels to be considered. Default is None.
    - test (bool, optional): Whether to evaluate the model on the test set. Default is True.
    - RMSE (bool, optional): Whether to calculate and save RMSE in the results. Default is True.
    - learning_rate (float, optional): Learning rate for the optimizer. Default is 0.001.
    - device (torch.device, optional): The device to use for training. Default is the global device.

    Returns:
    None: The results are saved to the specified CSV file.
    """
    # Create a path string
    directory_path = '..\\results'
    csv_file_path = os.path.join(directory_path, csv_file)
    tensorboard_path = os.path.join(directory_path, tensorboard_directory)

    # Initialize a dict to store the results
    results = {}
    # Loop over the configurations
    for hidden_layers in hidden_layers_options:
        for hidden_units in hidden_units_options:
            if weight_decay_options != None:
                data_noise = None
                for weight_decay in weight_decay_options:
                    if model_name == 'batch':
                        model = BatchEnsemble(ensemble_size=ensemble_size, input_shape=input_shape, hidden_layers=hidden_layers, hidden_units=hidden_units)
                    else:
                        # Shall implement for Naive model
                        model = None
                    # Train the model and get the average loss on test data
                    GNLLL_result, rmse_result, ENCE_result, epoch, ensemble_GNLLL_result, ensemble_rmse_result, ensemble_ence, train_time = train_models_w_mean_var(
                        model=model,
                        ensemble_type=model_name,
                        ensemble_size=ensemble_size,
                        data_noise=data_noise,
                        epochs=epochs,
                        print_frequency=print_frequency,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        test=test,
                        early_stopping=early_stopping,
                        RMSE=RMSE,
                        ENCE=ENCE,
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        device=device
                    )
                    # Save result in a dict
                    results = {
                        'model': model_name,
                        'ensemble_size': ensemble_size,
                        'hidden_layers': hidden_layers,
                        'hidden_units': hidden_units,
                        'weight_decay': weight_decay,
                        'data_noise': data_noise,
                        'epochs': epoch, # Epoch which model stopped training 
                        'optimizer': 'Adam',
                        'loss_fn': loss_fn.__class__.__name__,
                        'learning_rate': learning_rate,
                        'ENCE': ENCE_result,
                        'ensemble_ENCE': ensemble_ence,
                        'GNLLL': GNLLL_result,
                        'ensemble_GNLLL': ensemble_GNLLL_result,
                        'RMSE': rmse_result,
                        'ensemble_RMSE': ensemble_rmse_result,
                        'train_time': train_time
                    }
                    # Convert the results to a DataFrame
                    df_results = pd.DataFrame([results])

                    # Check if the CSV file exists
                    try:
                        # Load the existing CSV file
                        df_existing = pd.read_csv(csv_file_path)
                    except FileNotFoundError:
                        # If the file doesn't exist, create a new DataFrame
                        df_existing = pd.DataFrame()

                    # Save the combined DataFrame to the CSV file without rewriting the header
                    df_results.to_csv(csv_file_path, mode='a', header=not df_existing.shape[0], index=False)
            # If not weight decay, we are using AnchoredBatch
            else:
                model = AnchoredBatchEnsemble(ensemble_size=ensemble_size, input_shape=input_shape, hidden_layers=hidden_layers, hidden_units=hidden_units)
                weight_decay = None 
                for data_noise in data_noise_options:
                    comment = f'model={model_name} hidden_units={hidden_units} hidden_layers={hidden_layers} data_noise={data_noise} epochs={epochs}'
                    writer = SummaryWriter(log_dir=tensorboard_path,comment=comment)
                    # Train the model and get the average loss on test data
                    GNLLL_result, rmse_result, ENCE_result, epoch, ensemble_GNLLL_result, ensemble_rmse_result, ensemble_ence, train_time = train_models_w_mean_var(
                        model=model,
                        ensemble_type=model_name,
                        ensemble_size=ensemble_size,
                        data_noise=data_noise,
                        epochs=epochs,
                        print_frequency=print_frequency,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        test=test,
                        early_stopping=early_stopping,
                        RMSE=RMSE,
                        ENCE=ENCE,
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        writer=writer,
                        device=device
                    )

                    writer.add_hparams({'data_noise': data_noise, 'h_units': hidden_units, 'h_layers': hidden_layers, 'epochs':epochs}, 
                                       {'GNLLLoss':GNLLL_result, 'Ens.GNLLLoss': ensemble_GNLLL_result, 'RMSE': rmse_result, 'Ens.RMSE': ensemble_rmse_result, 
                                        'ENCE': ENCE_result, 'Ens.ENCE': ensemble_ence})
                    writer.close()
                    # Save result in a dict
                    results = {
                        'model': model_name,
                        'ensemble_size': ensemble_size,
                        'hidden_layers': hidden_layers,
                        'hidden_units': hidden_units,
                        'weight_decay': weight_decay,
                        'data_noise': data_noise,
                        'epochs': epoch, # Epoch which model stopped training 
                        'optimizer': 'Adam',
                        'loss_fn': loss_fn.__class__.__name__,
                        'learning_rate': learning_rate,
                        'ENCE': ENCE_result,
                        'ensemble_ENCE': ensemble_ence,
                        'GNLLL': GNLLL_result,
                        'ensemble_GNLLL': ensemble_GNLLL_result,
                        'RMSE': rmse_result,
                        'ensemble_RMSE': ensemble_rmse_result,
                        'train_time': train_time
                    }

                    # Convert the results to a DataFrame
                    df_results = pd.DataFrame([results])

                    # Check if the CSV file exists
                    try:
                        # Load the existing CSV file
                        df_existing = pd.read_csv(csv_file_path)
                    except FileNotFoundError:
                        # If the file doesn't exist, create a new DataFrame
                        df_existing = pd.DataFrame()

                    # Save the combined DataFrame to the CSV file without rewriting the header
                    df_results.to_csv(csv_file_path, mode='a', header=not df_existing.shape[0], index=False)


