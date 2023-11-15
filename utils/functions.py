# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
 
def train_models():
    return

def train_batch_ensemble():
    return
