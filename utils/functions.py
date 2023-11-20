# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union
import torch
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
    learning_rate: float,
    epochs: int,
    print_frequency: int,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device = device) -> None:
    """
    Train neural network models using various ensemble strategies.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        ensemble_type (str): The ensemble strategy ('batch', 'anchored_batch', 'naive').
        ensemble_size (int): Number of ensemble members (relevant for 'batch' and 'anchored_batch' ensembles).
        data_noise (float): Magnitude of noise to be added to the data for 'anchored_batch' ensemble.
        learning_rate (float): Learning rate for optimization.
        epochs (int): Number of training epochs.
        print_frequency (int): Frequency of printing training progress.
        data_loader (torch.utils.data.DataLoader): DataLoader providing training data.
        loss_fn (torch.nn.Module): Loss function for optimization.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        device (torch.device, optional): Device for training (default is device).

    Returns:
        None
    """
    if ensemble_type == 'batch' or ensemble_type == 'anchored_batch':
        optimizer = optimizer(model.parameters(), lr = learning_rate)
        model.to(device)
        for epoch in range(epochs):
            # Training
            train_loss = 0
            # Add a loop to loop through the training batches
            for batch, (X, y) in enumerate(data_loader):
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
                for batch, (X, y) in enumerate(data_loader):
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

            


def train_models_w_mean_var(
    model: Union[torch.nn.Module, list[torch.nn.Module]],
    ensemble_type: str,
    ensemble_size: int,
    data_noise: float,
    learning_rate: float,
    epochs: int,
    print_frequency: int,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device = device) -> None:
    """
    Train neural network models using various ensemble strategies.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        ensemble_type (str): The ensemble strategy ('batch', 'anchored_batch', 'naive').
        ensemble_size (int): Number of ensemble members (relevant for 'batch' and 'anchored_batch' ensembles).
        data_noise (float): Magnitude of noise to be added to the data for 'anchored_batch' ensemble.
        learning_rate (float): Learning rate for optimization.
        epochs (int): Number of training epochs.
        print_frequency (int): Frequency of printing training progress.
        data_loader (torch.utils.data.DataLoader): DataLoader providing training data.
        loss_fn (torch.nn.Module): Loss function for optimization.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        device (torch.device, optional): Device for training (default is device).

    Returns:
        None
    """
    if ensemble_type == 'batch' or ensemble_type == 'anchored_batch':
        optimizer = optimizer(model.parameters(), lr=learning_rate)
        model.to(device)
        for epoch in range(epochs):
            # Training
            train_loss = 0
            # Add a loop to loop through the training batches
            for batch, (X, y) in enumerate(data_loader):
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
                for batch, (X, y) in enumerate(data_loader):
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

