# Import libraries
import sys
sys.path.append('..')
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Optional, Tuple
import torch
from torch import nn
import pandas as pd
from utils.models import BatchEnsemble, AnchoredBatchEnsemble
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

def plot_predictive_2(data: np.ndarray, means: np.ndarray, variances: np.ndarray, xs: np.ndarray, title: str = None) -> None:
    """
    Plots predictive mean and variance with uncertainty bands given arrays of means and variances.

    Parameters:
    - data (np.ndarray): Data points for scatter plot.
    - means (np.ndarray): Predicted means for each ensemble member.
    - variances (np.ndarray): Predicted variances for each ensemble member.
    - xs (np.ndarray): X-axis values for the trajectories.
    - title (str, optional): Title for the plot.

    Returns:
    None
    """
    sns.set_style('darkgrid')
    palette = sns.color_palette('colorblind')
    
    blue = sns.color_palette()[0]
    red = sns.color_palette()[3]

    plt.figure(figsize=(9., 7.))
    
    plt.plot(data[:, 0], data[:, 1], "o", color=red, alpha=0.7, markeredgewidth=1., markeredgecolor="k")
    
    # Calculate overall mean and standard deviation
    mu = np.mean(means, axis=0)
    sigma = np.sqrt(np.mean(variances, axis=0))  # Convert variance to standard deviation

    plt.plot(xs, mu, "-", lw=2., color=blue)
    plt.plot(xs, mu-3 * sigma, "-", lw=0.75, color=blue)
    plt.plot(xs, mu+3 * sigma, "-", lw=0.75, color=blue)
        
    plt.fill_between(xs, mu-3*sigma, mu+3*sigma, alpha=0.35, color=blue)

    plt.xlim([np.min(xs), np.max(xs)])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
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
        learning_rate (float): Learning rate for optimization.
        device (torch.device, optional): Device for training (default is device).

    Returns:
         Union[None, float]: If test is False or test_loader is None, returns None. 
                           If test is True and test_loader is provided, returns the average test loss.
    """
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
                y_pred = model(X.repeat(ensemble_size,1)) # Make prediction

                # 2. Calculate loss per batch
                loss = loss_fn(y_pred, y.repeat(ensemble_size,1)) # Calculate loss with MSE

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
                    y_pred_test = model(X_test.repeat(ensemble_size, 1))
                    test_loss += loss_fn(y_pred_test, y_test.repeat(ensemble_size, 1)).item()

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

def calculate_ence(
    var_and_mse:torch.Tensor,
    num_bins: int=10
):
    """
    Calculate the Expected Normalized Calibration Error (ENCE) for a regression calibration evaluation[^1^][1].

    Parameters:
    var_and_mse (torch.Tensor): A tensor of variances and mean squared errors.
    num_bins (int): The number of bins to use for the calculation.

    Returns:
    float: The ENCE value.
    """

    # Initialize the ENCE value
    calibration_error = 0

    # Get the total number of data points
    N = len(var_and_mse)

    # Get the minimum and maximum variances
    min_var = torch.min(var_and_mse[:,0])
    max_var = torch.max(var_and_mse[:,0])

    # Arrange bin ranges uniformly 
    bin_ranges = torch.linspace(min_var, max_var, num_bins)

    # Sort the values
    var_and_mse = torch.sort(var_and_mse,dim=0)[0]

    # Initialize the index for the sorted tensor
    bin_start_index = 0

    # Iterate over the bin ranges
    for max_var in bin_ranges[1:]:
        # Initialize the cardinality of the bin
        num_in_bin = 0

        # Initialize variables to store the mean squared error and variance of each bin
        bin_mse = 0
        bin_var = 0

        # Iterate over the variances and mean squared errors
        for var, mse in zip(var_and_mse[bin_start_index:,0], var_and_mse[bin_start_index:,1]):
            # Accumulate the mean squared error and variance of each bin, count the cardinality of the bin
            if var <= max_var:
                bin_mse += mse
                bin_var  += var
                num_in_bin += 1
                bin_start_index += 1
            else:
                # We have gone through all elements in the bin
                break
        
        # Ensure no division by zero
        if num_in_bin != 0:
            # Calculate the root mean square error and mean variance of each bin
            rmse = torch.sqrt(bin_mse / num_in_bin)
            mean_var = torch.sqrt(bin_var / num_in_bin)

            # Add each bin score to the final ENCE variable
            calibration_error += torch.abs(mean_var - rmse) / mean_var

    # Normalize the ENCE with the number of data points
    calibration_error /= N

    return calibration_error.item()

    

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
    learning_rate: Optional[float]=0.001,
    weight_decay: Optional[float]=None,
    device: torch.device = device) -> Optional[Union[float, Tuple[float, float]]]:
    """
    Train neural network models using various ensemble strategies.

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
        test (bool): True or False, Shall we measure on test data
        RMSE (bool): If True will calculate RMSE
        learning_rate (float): Learning rate for optimization.
        device (torch.device, optional): Device for training (default is device).

    Returns:
        Optional[Union[float, Tuple[float, float]]]: If `test` is False or `test_loader` is None, returns None.
        If `test` is True and `test_loader` is provided, returns the average test loss.
        If `RMSE` is True, additionally returns the Root Mean Squared Error (RMSE).
    """
    # If RMSE initiate MSE loss function
    if RMSE:
        MSE_loss_fn = nn.MSELoss(reduction='none')

    if ensemble_type == 'batch' or ensemble_type == 'anchored_batch':
        optimizer = optimizer(model.parameters(), lr=learning_rate)
        model.to(device)
        for epoch in range(epochs):
            # Training
            train_loss = 0
            # Add a loop to loop through the training batches
            for batch, (X, y) in enumerate(train_loader):
                model.train()
                # 1. Perform forward pass
                mean, var = model(X.repeat(ensemble_size, 1))  # Make prediction

                # 2. Calculate loss per batch
                loss = loss_fn(mean, y.repeat(ensemble_size, 1), var)  # Calculate loss with MSE

                train_loss += loss.item()  # Accumulate loss

                if ensemble_type == 'anchored_batch':
                    batch_size = X.shape[0]
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

            if epoch % print_frequency == 0:
                print(f"Epoch: {epoch}\n-------")
                print("Loss:", train_loss / (batch + 1))  # Calculate and print average loss per batch
        if test and test_loader is not None:
            model.eval()
            test_loss = 0
            # If True initiate MSE Loss
            if RMSE:
                rmse_loss = 0
                if ENCE:
                    # Initiate an empty tensor
                    var_mse = torch.Tensor().to(device)
            with torch.no_grad():
                for batch, (X_test, y_test) in enumerate(test_loader):
                    mean_test, var_test = model(X_test.repeat(ensemble_size, 1))
                    test_loss += loss_fn(mean_test, y_test.repeat(ensemble_size, 1), var_test).item()
                    # If True Calculate the RMSE
                    if RMSE:
                        MSE = MSE_loss_fn(mean_test, y_test.repeat(ensemble_size, 1))
                        #rmse_loss += torch.sqrt(MSE_loss_fn(mean_test, y_test.repeat(ensemble_size, 1))).item()
                        if ENCE:
                            # Gather the predicted standard deviation of model and its corresponding RMSE
                            batch_result = torch.cat((var_test, MSE), dim = 1)

                            # Append the batch result
                            var_mse = torch.cat((var_mse, batch_result), dim = 0)
                        rmse_loss += torch.sqrt(MSE.mean()).item()

            # Compute the average over the batches
            average_test_loss = test_loss / (batch + 1)
            print(f"\nEvaluation on Test Data\n------------------------")
            print("Average Test Loss:", average_test_loss)
            if RMSE:
                average_rmse_loss = rmse_loss  = rmse_loss / (batch + 1)
                if ENCE:
                    ence = calculate_ence(var_mse)
                    return average_test_loss, average_rmse_loss, ence
                return average_test_loss, average_rmse_loss
            else:
                return average_test_loss



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
                        mean_test, var_test = Model(X_test.repeat(ensemble_size, 1))
                        Model_test_loss += loss_fn(mean_test, y_test.repeat(ensemble_size, 1), var_test).item()
                        # If True calculate RMSE
                        if RMSE:
                            Model_rmse_loss += torch.sqrt(MSE_loss_fn(mean_test, y_test.repeat(ensemble_size, 1))).item()
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
        RMSE: bool = True,
        learning_rate: Optional[float] = 0.001,
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
    # Initialize a list to store the results
    results = []
    # ENCE to be implemented
    ENCE = None
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
                    GNLLL_result, rmse_result = train_models_w_mean_var(
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
                        RMSE=RMSE,
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        device=device
                    )
                    # Append the result to the list
                    results.append({
                        'model': model_name,
                        'ensemble_size': ensemble_size,
                        'hidden_layers': hidden_layers,
                        'hidden_units': hidden_units,
                        'weight_decay': weight_decay,
                        'data_noise': data_noise,
                        'epochs': epochs,
                        'optimizer': 'Adam',
                        'loss_fn': loss_fn.__class__.__name__,
                        'learning_rate': learning_rate,
                        'ENCE': ENCE,
                        'GNLLL': GNLLL_result,
                        'RMSE': rmse_result
                    })
            # If not weight decay, we are using AnchoredBatch
            else:
                model = AnchoredBatchEnsemble(ensemble_size=ensemble_size, input_shape=input_shape, hidden_layers=hidden_layers, hidden_units=hidden_units)
                weight_decay = None 
                for data_noise in data_noise_options:
                    # Train the model and get the average loss on test data
                    GNLLL_result, rmse_result = train_models_w_mean_var(
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
                        RMSE=RMSE,
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        device=device
                    )

                    # Append the result to the list
                    results.append({
                        'model': model_name,
                        'ensemble_size': ensemble_size,
                        'hidden_layers': hidden_layers,
                        'hidden_units': hidden_units,
                        'weight_decay': weight_decay,
                        'data_noise': data_noise,
                        'epochs': epochs,
                        'optimizer': 'Adam',
                        'loss_fn': loss_fn.__class__.__name__,
                        'learning_rate': learning_rate,
                        'ENCE': ENCE,
                        'GNLLL': GNLLL_result,
                        'RMSE': rmse_result
                    })

    # Convert the results to a DataFrame
    df_results = pd.DataFrame(results)

    # Create a path string
    directory_path = '..\\results'
    csv_file_path = os.path.join(directory_path, csv_file)

    # Check if the CSV file exists
    try:
        # Load the existing CSV file
        df_existing = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        # If the file doesn't exist, create a new DataFrame
        df_existing = pd.DataFrame()

    # Save the combined DataFrame to the CSV file without rewriting the header
    df_results.to_csv(csv_file_path, mode='a', header=not df_existing.shape[0], index=False)
