import torch
from scipy.stats import norm

def compute_z_score(pred_interval: float) -> float:
    """
    Compute the z-score based on a given prediction interval.

    Parameters:
    - pred_interval (float): Prediction interval in the range (0, 100).

    Returns:
    float: The z-score corresponding to the given prediction interval.
    """
    # Convert pred_interval to a probability
    probability = pred_interval / 100

    # Compute z-score based on the probability
    z = norm.ppf((1 + probability) / 2)
    
    return z

def calculate_PIC_PIW(PIC:float, PIW:float, n:int, means:torch.Tensor, variances: torch.Tensor, y_true: torch.Tensor, pred_interval: float=90.0):
    """"
    Calculate Prediction Interval Coverage (PIC) and Prediction Interval Width (PIW) for a batch of predictions.

    Parameters:
    - PIC (float): Current Prediction Interval Coverage.
    - PIW (float): Current Prediction Interval Width.
    - n (int): Current total number of data points.
    - means (torch.Tensor): Tensor containing mean predictions.
    - variances (torch.Tensor): Tensor containing variances of predictions.
    - y_true (torch.Tensor): Tensor containing true values.
    - pred_interval (float, optional): Desired prediction interval (default is 90.0).

    Returns:
    Tuple of three values (PIC, PIW, n) representing the updated values after processing the current batch.

    This function calculates the Prediction Interval Coverage (PIC) and Prediction Interval Width (PIW) for a batch
    of predictions.

    The Prediction Interval (PI) is calculated as [mean - z * sqrt(variance), mean + z * sqrt(variance)], where z is
    determined based on the desired prediction interval.

    The function updates the input PIC, PIW, and n based on the performance of the model on the current batch.
    """
    z = compute_z_score(pred_interval)
    sigma = torch.sqrt(variances)
    # Update n by the batch size
    n += means.shape[0]
    # Calculate y_pred lower bound
    y_pred_L = means-(sigma*z) 
    # Calculate y_pred upper bound
    y_pred_U = means+(sigma*z)
    # Check if  the true values are within the prediction interval
    above_lower_bound = y_pred_L < y_true
    below_upper_bound = y_pred_U > y_true
    # Calculate total number of data points captured for current batch and update
    # the Prediction Interval Coverage
    captured_points = (above_lower_bound*below_upper_bound).sum().item()
    PIC += (above_lower_bound*below_upper_bound).sum().item()
    # Calculate the current batch's PI width and update the Prediction Interval Width (PIW)
    PIW += (y_pred_U-y_pred_L).sum().item()
    # Update PICP
    return PIC,PIW, n

def calculate_ence(
    var_and_mse:torch.Tensor,
    num_bins: int=10
):
    """
    NOT WORKING PROPERLY
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
