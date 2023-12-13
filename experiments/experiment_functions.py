import sys
sys.path.append('..')
import torch
from torch import nn
import copy
from torch.utils.data import TensorDataset, DataLoader
from typing import Union, Optional, Tuple
from sklearn.model_selection import train_test_split
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from utils.functions import calculate_uncertainties
from utils.models import KaimingNN
from utils.metrics import calculate_PIC_PIW
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_datasets(test_size:float=0.1, val_size:float=0.2, batch_size:int=128):
    """
    Prepare PyTorch DataLoader instances for power and concrete datasets.

    Parameters:
    - test_size (float): Proportion of the dataset to include in the test split.
    - val_size (float): Proportion of the training dataset to include in the validation split.
    - batch_size (int): Number of samples per batch to load into DataLoader.

    Returns:
    - power_train_loader, power_val_loader, power_test_loader,
      concrete_train_loader, concrete_val_loader, concrete_test_loader:
      DataLoader instances for training, validation, and test sets for both power and concrete datasets.
    """
    # Read data into a pandas dataframe
    power = pd.read_excel('..\\data\\UCI_Regression\\5.PowerPlant\\Folds5x2_pp.xlsx')
    concrete = pd.read_excel('.\\data\\UCI_Regression\\2.Concrete\\Concrete_Data.xls')

    # Perform first split, training and test set
    power_train, power_test = train_test_split(power, test_size=test_size)
    power_train, power_test = power_train.to_numpy(), power_test.to_numpy()
    concrete_train, concrete_test = train_test_split(concrete, test_size=test_size)
    concrete_train, concrete_test = concrete_train.to_numpy(), concrete_test.to_numpy()

    # Split the training datasets into features and labels
    X_power_train = power_train[:, :4]
    Y_power_train = power_train[:, 4]
    X_concrete_train = concrete_train[:, :8]
    Y_concrete_train = concrete_train[:, 8]

    X_power_test = power_test[:, :4]
    Y_power_test = power_test[:, 4]
    X_concrete_test = concrete_test[:, :8]
    Y_concrete_test = concrete_test[:, 8]

    # Split the training dataset into a training and validation datset
    x_power_train, x_power_val, y_power_train, y_power_val = train_test_split(X_power_train, Y_power_train, test_size=val_size)
    x_concrete_train, x_conrete_val, y_concrete_train, y_concrete_val = train_test_split(X_concrete_train, Y_concrete_train, test_size=val_size)

    # Normalise data by subtracting mean and divide by standard deviation
    standard_scaler = StandardScaler()
    x_power_train = standard_scaler.fit_transform(x_power_train)
    x_power_val = standard_scaler.fit_transform(x_power_val)
    X_power_test = standard_scaler.fit_transform(X_power_test)

    x_concrete_train = standard_scaler.fit_transform(x_concrete_train)
    x_concrete_val = standard_scaler.fit_transform(x_concrete_val)
    X_conrete_test = standard_scaler.fit_transform(X_concrete_test)

    # Convert numpy arrays to PyTorch Tensors
    x_power_train_tensor = torch.tensor(x_power_train, dtype=torch.float32).to(device)
    y_power_train_tensor = torch.tensor(y_power_train, dtype=torch.float32).unsqueeze(1).to(device)
    x_power_val_tensor = torch.tensor(x_power_val, dtype=torch.float32).to(device)
    y_power_val_tensor = torch.tensor(y_power_val, dtype=torch.float32).unsqueeze(1).to(device)
    x_power_test_tensor = torch.tensor(X_power_test, dtype=torch.float32).to(device)
    y_power_test_tensor = torch.tensor(Y_power_test, dtype=torch.float32).unsqueeze(1).to(device)

    x_concrete_train_tensor = torch.tensor(x_concrete_train, dtype=torch.float32).to(device)
    y_concrete_train_tensor = torch.tensor(y_concrete_train, dtype=torch.float32).unsqueeze(1).to(device)
    x_concrete_val_tensor = torch.tensor(x_concrete_val, dtype=torch.float32).to(device)
    y_concrete_val_tensor = torch.tensor(y_concrete_val, dtype=torch.float32).unsqueeze(1).to(device)
    x_concrete_test_tensor = torch.tensor(X_concrete_test, dtype=torch.float32).to(device)
    y_concrete_test_tensor = torch.tensor(Y_concrete_test, dtype=torch.float32).unsqueeze(1).to(device)

    # Create TensorDatasets for power dataset
    power_train_dataset = TensorDataset(x_power_train_tensor, y_power_train_tensor)
    power_val_dataset = TensorDataset(x_power_val_tensor, y_power_val_tensor)
    power_test_dataset = TensorDataset(x_power_test_tensor, y_power_test_tensor)

    # Create TensorDatasets for concrete dataset
    concrete_train_dataset = TensorDataset(x_concrete_train_tensor, y_concrete_train_tensor)
    concrete_val_dataset = TensorDataset(x_concrete_val_tensor, y_concrete_val_tensor)
    concrete_test_dataset = TensorDataset(x_concrete_test_tensor, y_concrete_test_tensor)


    # Create DataLoaders for power dataset
    power_train_loader = DataLoader(dataset=power_train_dataset, batch_size=batch_size, shuffle=True)
    power_val_loader = DataLoader(dataset=power_val_dataset, batch_size=batch_size, shuffle=False)
    power_test_loader = DataLoader(dataset=power_test_dataset, batch_size=batch_size, shuffle=False)

    # Create DataLoaders for concrete dataset
    concrete_train_loader = DataLoader(dataset=concrete_train_dataset, batch_size=batch_size, shuffle=True)
    concrete_val_loader = DataLoader(dataset=concrete_val_dataset, batch_size=batch_size, shuffle=False)
    concrete_test_loader = DataLoader(dataset=concrete_test_dataset, batch_size=batch_size, shuffle=False)

    return power_train_loader, power_val_loader, power_test_loader, concrete_train_loader, concrete_val_loader, concrete_test_loader

def train_ensemble_model(
    model: torch.nn.Module,
    ensemble_type: str,
    ensemble_size: int,
    epochs: int,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader],
    data_noise: Optional[float],
    input_shape: Optional[int],
    h_layers: Optional[int],
    h_units: Optional[int],
    weight_decay: Optional[float],
    learning_rate: Optional[float]=0.001   
):
    """
    This function is used for the experiments.
    Trains an ensemble model, returns best trained model and training time
    """
    # Record the start time for training
    start_time_training = time.time()

    if ensemble_type == 'batch' or ensemble_type == 'anchored_batch':
         # Initiate a variable to track best model performance
        best_loss = None
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

            with torch.no_grad():
                test_loss = 0
                for batch, (X_val, y_val) in enumerate(val_loader):
                    mean_test, var_val = model(X_val.unsqueeze(1).expand(-1, ensemble_size, -1))
                    test_loss += loss_fn(mean_test, y_val.unsqueeze(1).expand(-1, ensemble_size, -1), var_val).item()
                average_test_loss = test_loss / (batch + 1)
                if best_loss is None or average_test_loss < best_loss:
                    # Update model's best performance
                    best_loss = average_test_loss
                    # Save the model's weights w/ best performance
                    best_model = copy.deepcopy(model.state_dict())

        # Update model with the weights with the best performance
        final_model = model.load_state_dict(best_model)

    elif ensemble_type == 'naive':
        # Initialize NNs and store them in a list
        NaiveEnsemble = [KaimingNN(input_shape=input_shape,
                                   hidden_layers=h_layers,
                                   hidden_units=h_units,
                                   ) 
                                   for i in range(ensemble_size)]
        # Train each Ensemble member
        for model in NaiveEnsemble:
            # Initialize best loss as None for each model
            best_loss = None
            # We're optimizing each model's parameters
            model_optimizer = optimizer(model.parameters(), lr= learning_rate)
            model.to(device)

            # Conduct training
            for epoch in range(epochs):
                # Add a loop to loop through the training batches
                for batch, (X, y) in enumerate(train_loader):
                    model.train()
                    # 1. Perform forward pass
                    mean_pred, var_pred = model(X) # Make prediction

                    # 2. Calculate loss per batch
                    loss = loss_fn(mean_pred, y, var_pred) 

                    # 3. Optimizer zero grad
                    optimizer.zero_grad() # Set the optimizer's gradients to zero

                    # 4. Loss backward
                    loss.backward()

                    #5. Optimizer step
                    optimizer.step()
                
                # Test model on validation data
                test_loss = 0
                with torch.no_grad():
                    for batch, (X_val, y_val) in enumerate(val_loader):
                        # Make prediction
                        mean_pred, var_pred = model(X_val) # Make prediction
                        # Calculate loss per batch
                        loss = loss_fn(mean_pred, y_val, var_pred) 
                        # Acumalate loss per batch
                        test_loss += loss.item()
                    # Calculate the average test loss accross the batches
                    average_test_loss = test_loss / (batch+1)

                    if best_loss is None or average_test_loss < best_loss:
                        # Update model's best performance
                        best_loss = average_test_loss
                        # Save the model's weights w/ best performance
                        best_model = copy.deepcopy(model.state_dict())
            
            # Keep the best performing model weights
            model = model.load_state_dict(best_model)
        # Once we have trained all ensemble members we have a final ensemble model
        final_model = NaiveEnsemble
    end_time_training = time.time()
    # Calculate training time
    training_time = end_time_training-start_time_training 

    return final_model, training_time


def test_model(model,
               model_name,
               test_loader,
               ensemble_size: Optional[int],
               batch_size: Optional[int],
               test_time_iterations: Optional[int] = 10):
    model.eval()
    test_loss = 0
    # Initialize variables for calculating PICP and MPIW
    n = 0
    pic = 0.0
    piw = 0.0
    rmse_loss = 0
    MSE_loss_fn = nn.MSELoss()

    if model_name == 'anchored_batch' or model_name == 'batch':
        with torch.no_grad():
            for batch, (X_test, y_test) in enumerate(test_loader):
                mean_test, var_test = model(X_test.unsqueeze(1).expand(-1, ensemble_size, -1))
                
                # Calculate combined ensemble's loss
                aleatoric_uncertainty, epistemic_uncertainty, combined_uncertainty = calculate_uncertainties(mean_test,var_test,ensemble_size)
                ensemble_mean = torch.mean(mean_test, dim=1)

                MSE = MSE_loss_fn(ensemble_mean, y_test)
                rmse_loss += torch.sqrt(MSE).item()
                pic,piw, n = calculate_PIC_PIW(pic,piw, n, ensemble_mean, combined_uncertainty, y_test)


    elif model_name == 'naive':
        with torch.no_grad():
            for batch, (X_test, y_test) in enumerate(test_loader):
                # Initialize tensors which will collect mean and variance predictions/batch from all ensemble members
                ensemble_mean, ensemble_var = torch.zeros((batch_size,ensemble_size,1)), torch.zeros((batch_size,ensemble_size,1))
                for ensemble_member, index in zip(model, len(model)):
                    member_mean, member_var = ensemble_member(X_test)
                    # Gather each ensemble member's prediction
                    ensemble_mean[:,ensemble_size,:], ensemble_var[:,ensemble_size,:] = member_mean, member_var
                
                # Calculate combined ensemble's mean & variance (combined uncertainty considered variance)
                aleatoric_uncertainty, epistemic_uncertainty, combined_uncertainty = calculate_uncertainties(ensemble_mean,ensemble_var,ensemble_size)
                ensemble_mean = torch.mean(ensemble_mean, dim=1)

                # Calculate metrics using ensemble's combined prediction
                MSE = MSE_loss_fn(ensemble_mean, y_test)
                rmse_loss += torch.sqrt(MSE).item()
                pic, piw, n = calculate_PIC_PIW(pic, piw, n, ensemble_mean, combined_uncertainty, y_test)
    
    # Calculate Average RMSE across batches
    average_rmse_loss = rmse_loss / (batch + 1)
    # Calculate Prediction Interval Coverage Probabiblity (PICP)
    PICP = pic/n
    # Calculate Mean Prediction Interval Width (MPIW)
    MPIW = piw/n
    return average_rmse_loss, PICP, MPIW
            
