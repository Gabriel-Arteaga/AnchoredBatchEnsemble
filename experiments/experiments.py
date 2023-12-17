# Import dependencies 
import os
import sys
sys.path.append('..')
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from itertools import product
from utils.models import AnchoredBatchEnsemble, BatchEnsemble
from GP_functions import model_all_points, model_inducing_half, test_GP_inference_time
from experiment_functions import *

# Inputs
epochs = 1000
n_iterations = 10
batch_size = 128
test_size = 0.1
val_size = 0.2
input_shape_power = 4
input_shape_concrete = 8

# Second Experiment inputs
ensemble_sizes = [1,2,4,8,16,32]
second_experiment_models = ['anchored_batch', 'naive']

# Create a path string
directory_path = '..\\results'
csv_file = 'experiment_results.csv'

# NN Ensemble Hyperparameters for power dataset
n_hidden_layers_power = 3
n_hidden_units_power = 256
ensemble_size_power = 8
data_noise_power = 1e-4
weight_decay_power = 1e-3
dropout_prob_power = 0

# NN Ensemble Hyperparameters  for concrete dataset
n_hidden_layers_concrete = 6
n_hidden_units_concrete = 128
ensemble_size_concrete = 16
data_noise_concrete = 1e-4
weight_decay_concrete = 1e-6
dropout_prob_concrete = 0

models = ['anchored_batch', 'batch', 'naive', 'GP', 'GP_induce']
datasets = ['power', 'concrete']

# We train and test the models 10 times, randomly splitting the data each time
    # Prepare dataloaders for training and testing, both power and concrete dataset
power_train_loader, power_val_loader, power_test_loader, concrete_train_loader, concrete_val_loader, concrete_test_loader,\
     x_train_power, y_train_power, x_test_power, y_test_power, x_train_concrete, y_train_concrete, x_test_concrete, y_test_concrete =\
        prepare_datasets(test_size=test_size, val_size=val_size, batch_size=batch_size)

for dataset_name, model_name in product(datasets, models):
    print(f'Current dataset:{dataset_name} Current model: {model_name}')
    if dataset_name == 'power':
        if model_name == 'anchored_batch':
            model = AnchoredBatchEnsemble(ensemble_size=ensemble_size_power,
                                            input_shape=input_shape_power,
                                            hidden_layers=n_hidden_layers_power, 
                                            hidden_units= n_hidden_units_power,
                                            dropout_prob = dropout_prob_power
                                            )
        elif model_name == 'batch':
            model = BatchEnsemble(ensemble_size=ensemble_size_power,
                                input_shape=input_shape_power,
                                hidden_layers=n_hidden_layers_power,
                                hidden_units= n_hidden_units_power
                                )
        elif model_name == 'naive':
            # Will need to be created and initialized inside training loop
            model = None

        elif model_name == 'GP':
            training_time, RMSE, PICP, MPIW, model, _ = model_all_points(x_train_power, y_train_power, x_test_power, y_test_power, epochs=epochs)
        
        elif model_name == 'GP_induce':
            training_time, RMSE, PICP, MPIW, model, _ = model_inducing_half(x_train_power, y_train_power, x_test_power, y_test_power, epochs=epochs)

        if model_name == 'batch' or model_name == 'anchored_batch' or model_name=='naive':   
            model, training_time = train_ensemble_model(model=model, 
                                    ensemble_type=model_name,
                                    ensemble_size=ensemble_size_power, 
                                    data_noise=data_noise_power,
                                    loss_fn=nn.GaussianNLLLoss(),
                                    epochs=epochs,
                                    optimizer=torch.optim.Adam,
                                    train_loader=power_train_loader,
                                    val_loader=power_val_loader,
                                    input_shape = input_shape_power,
                                    h_layers = n_hidden_layers_power,
                                    h_units = n_hidden_units_power,
                                    weight_decay=weight_decay_power,
                                    )
            
    
    elif dataset_name == 'concrete':
        if model_name == 'anchored_batch':
            model = AnchoredBatchEnsemble(ensemble_size=ensemble_size_concrete, input_shape=input_shape_concrete,
                                            hidden_layers=n_hidden_layers_concrete, hidden_units= n_hidden_units_concrete,
                                            dropout_prob = dropout_prob_concrete)
            
        elif model_name == 'batch':
            model = BatchEnsemble(ensemble_size=ensemble_size_concrete,
                                    input_shape=input_shape_concrete,
                                    hidden_layers=n_hidden_layers_concrete,
                                    hidden_units= n_hidden_units_concrete
                                    )

        elif model_name == 'GP':
            training_time, RMSE, PICP, MPIW, model,_ = model_all_points(x_train_concrete, y_train_concrete, x_test_concrete, y_test_concrete, epochs=epochs)
        
        elif model_name == 'GP_induce':
            training_time, RMSE, PICP, MPIW, model, _ = model_inducing_half(x_train_concrete, y_train_concrete, x_test_concrete, y_test_concrete, epochs=epochs)

        if model_name == 'batch' or model_name == 'anchored_batch' or model_name == 'naive':
            model, training_time = train_ensemble_model(model=model, 
                                                            ensemble_type=model_name,
                                                            ensemble_size=ensemble_size_concrete, 
                                                            data_noise=data_noise_concrete,
                                                            epochs=epochs,
                                                            loss_fn=nn.GaussianNLLLoss(),
                                                            optimizer=torch.optim.Adam,
                                                            train_loader=concrete_train_loader,
                                                            val_loader=concrete_val_loader,
                                                            input_shape = input_shape_concrete,
                                                            h_layers = n_hidden_layers_concrete,
                                                            h_units = n_hidden_units_concrete,
                                                            weight_decay=weight_decay_concrete,
                                                            )
    
    # We test the models performance in terms of RMSE, PICP, MPIW
    if dataset_name == 'concrete':
        if model_name == 'anchored_batch' or model_name=='batch' or model_name == 'naive':
            RMSE, PICP, MPIW = test_model(model=model,
                                        model_name=model_name,
                                        test_loader=concrete_test_loader,
                                        ensemble_size = ensemble_size_concrete
                                        )
            inference_time = test_inference_time(model=model,
                                                 model_name=model_name,
                                                 test_loader=concrete_test_loader,
                                                 ensemble_size=ensemble_size_concrete,
                                                 n_iterations=n_iterations)
        else:
            inference_time = test_GP_inference_time(model, x_test_concrete, n_iterations=n_iterations)
    elif dataset_name == 'power':
        if model_name == 'anchored_batch' or model_name=='batch' or model_name == 'naive':
            RMSE, PICP, MPIW = test_model(model=model,
                                        model_name=model_name,
                                        test_loader=power_test_loader,
                                        ensemble_size = ensemble_size_power
                                        )
            inference_time = test_inference_time(model=model,
                                                 model_name=model_name,
                                                 test_loader=power_test_loader,
                                                 ensemble_size=ensemble_size_power,
                                                 n_iterations=n_iterations)
        else:
            inference_time = test_GP_inference_time(model, x_test_power, n_iterations=n_iterations)

                                      
                    
    # Save result in a dict
    csv_file_path = os.path.join(directory_path, csv_file)
    results = {
                'dataset': dataset_name,
                'model': model_name,
                'RMSE': RMSE,
                'PICP': PICP,
                'MPIW': MPIW,
                'train_time': training_time,
                'inference_time': inference_time
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

# Start second experiment
for model_name, ensemble_size in product(second_experiment_models, ensemble_sizes):
    if model_name == 'anchored_batch':
        model = AnchoredBatchEnsemble(ensemble_size=ensemble_size_concrete, input_shape=input_shape_concrete,
                                            hidden_layers=n_hidden_layers_concrete, hidden_units= n_hidden_units_concrete,
                                            dropout_prob = dropout_prob_concrete)
    else:
        model = None 
    model, training_time = train_ensemble_model(model=model, 
                                                            ensemble_type=model_name,
                                                            ensemble_size=ensemble_size_concrete, 
                                                            data_noise=data_noise_concrete,
                                                            epochs=epochs,
                                                            loss_fn=nn.GaussianNLLLoss(),
                                                            optimizer=torch.optim.Adam,
                                                            train_loader=concrete_train_loader,
                                                            val_loader=concrete_val_loader,
                                                            input_shape = input_shape_concrete,
                                                            h_layers = n_hidden_layers_concrete,
                                                            h_units = n_hidden_units_concrete,
                                                            weight_decay=weight_decay_concrete,
                                                            )

    # Save result in a dict
    csv_file_path = os.path.join(directory_path, 'training_times.csv')
    results = {
                'model': model_name,
                'ensemble_size':ensemble_size,
                'train_time': training_time
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