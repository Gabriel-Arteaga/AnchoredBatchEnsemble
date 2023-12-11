# Import dependencies 
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
from utils.functions import prepare_datasets, train_ensemble_model

# Inputs
batch_size = 128
test_size = 0.1
val_size = 0.2
input_shape_power = 4
input_shape_concrete = 8

# NN Ensemble Hyperparameters for power dataset
n_hidden_layers_power = X
n_hidden_units_power = X
ensemble_size_power = X
data_noise_power = X
weight_decay_power = X
dropout_prob_power = X

# NN Ensemble Hyperparameters  for concrete dataset
n_hidden_layers_concrete = X
n_hidden_units_concrete = X
ensemble_size_concrete = X
data_noise_conrete = X
weight_decay_concrete = X
dropout_prob_concrete = X

models = ['anchored_batch', 'batch', 'GP', 'GP_induce']
datasets = ['power', 'concrete']

# We train and test the models 10 times, randomly splitting the data each time
for run in range(10):
    # Prepare dataloaders for training and testing, both power and concrete dataset
    power_train_loader, power_val_loader, power_test_loader, concrete_train_loader, concrete_val_loader, concrete_test_loader =\
          prepare_datasets(test_size=test_size, val_size=val_size, batch_size=batch_size)
    
    for dataset_name, model_name in product(datasets, models):
        if dataset_name == 'power':
            if model_name == 'anchored_batch':
                model = AnchoredBatchEnsemble(ensemble_size=ensemble_size_power,
                                                input_shape=input_shape_power,
                                                hidden_layers=n_hidden_layers_power, 
                                                hidden_units= n_hidden_units_power,
                                                dropout_pob = dropout_prob_power
                                                )
            elif model_name == 'batch':
                model = BatchEnsemble(ensemble_size=ensemble_size_power,
                                    input_shape=input_shape_power,
                                    hidden_layers=n_hidden_layers_power,
                                    hidden_units= n_hidden_units_power,
                                    dropout_pob = dropout_prob_power
                                    )
                
            elif model_name == 'GP':
                model = X # Alex fixes GP init. for power dataset
            
            elif model_name == 'GP_induce':
                model = X # Alex fixes GP w/ inducing points init. for power dataset

            if model_name == 'batch' or model_name = 'anchored_batch':   
                model, training_time = train_ensemble_model(model=model, 
                                        ensemble_type=model_name,
                                        ensemble_size=ensemble_size_power, 
                                        data_noise=data_noise_power,
                                        loss_fn=nn.GaussianNLLLoss(),
                                        optimizer=torch.optim.Adam,
                                        train_loader=power_train_loader,
                                        val_loader=power_val_loader
                                        )
            else:
                pass # Alex fixes training loop which trains model an measures training time
        
        elif dataset_name == 'concrete':
            if model_name == 'anchored_batch':
                model = AnchoredBatchEnsemble(ensemble_size=ensemble_size_concrete, input_shape=input_shape_concrete,
                                                hidden_layers=n_hidden_layers_concrete, hidden_units= n_hidden_units_concrete,
                                                dropout_pob = dropout_prob_concrete)
                
            elif model_name == 'batch':
                model = BatchEnsemble(ensemble_size=ensemble_size_concrete,
                                        input_shape=input_shape_concrete,
                                        hidden_layers=n_hidden_layers_concrete,
                                        hidden_units= n_hidden_units_concrete,
                                        dropout_pob = dropout_prob_concrete
                                        )

            elif model_name == 'GP':
                model = X # Alex fixes GP models
            
            elif model_name == 'GP_induce':
                model = X # Alex

            if model_name == 'batch' or model_name == 'anchored_batch':
                model, training_time = train_ensemble_model(model=model, 
                                                                ensemble_type=model_name,
                                                                ensemble_size=ensemble_size_concrete, 
                                                                data_noise=data_noise_concrete,
                                                                loss_fn=nn.GaussianNLLLoss(),
                                                                optimizer=torch.optim.Adam,
                                                                train_loader=concrete_train_loader,
                                                                val_loader=concrete_val_loader
                                                                )
            else:
                pass # Alex fixes training loop which trains model an measures training time
        
        # We test the models performance in terms of RMSE, PICP, MPIW and inference time
        RMSE, PICP, MPIW, inference_time = test_model()
        # Save result in a dict
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
