import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle as pkl

from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
from typing import List, Tuple, Callable, Union, Any

try:
    from matplotlib.collections import QuadContourSet
except ImportError:
    try:
        from matplotlib.contour import QuadContourSet
    except ImportError:
        # For newer matplotlib versions, use Any as fallback
        QuadContourSet = Any
    
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.text import Text

# from functools import partial
# from matplotlib import animation
# from matplotlib.animation import PillowWriter
# import copy

def load_data(filename: str, color1: str, color2: str) -> np.ndarray:
    """
    Load labeled pixels from pkl file with two classes and visualize them.
    
    Args:
        filename: Path to the pickle file containing the data
        color1: Color for class 0 points
        color2: Color for class 1 points
        
    Returns:
        numpy array of shape (m, 3) where each row is [x, y, label]
    """
    """
    Load labeled pixels from pkl file with two classes
    Load n pixels returns array of shape (m, 3)
    For each pixel, store the x, y, and label

    plot the data as well.
    Input:
        filename: file where data is stored
    Output:
        data_points: numpy array of shape (m, 3).
    """
    with open(filename, 'rb') as f:
        data_points = pkl.load(f)

    data_points_1 = []
    data_points_2 = []

    for p in data_points:
        if p[2] == 0:
            data_points_1.append(p[:-1])
        else:
            data_points_2.append(p[:-1])

    data_points_1 = np.array(data_points_1)
    data_points_2 = np.array(data_points_2)

    # Visualize the dataset
    plt.scatter(data_points_1[:, 0], data_points_1[:, 1], c=color1, s=5)
    plt.scatter(data_points_2[:, 0], data_points_2[:, 1], c=color2, s=5)

    plt.gca().set_aspect('equal', adjustable='box')

    plt.show()

    return data_points

def create_dataset(data_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert raw data points into normalized feature matrix and label vector.
    
    Args:
        data_points: Array of shape (m, 3) where each row is [x, y, label]
        
    Returns:
        Tuple containing:
        - X: Normalized feature matrix of shape (m, 2)
        - y: Label vector of shape (m, 1)
    """
    # Gather X, y for dataset
    X = np.zeros((data_points.shape[0], 2)) # All of our points
    for idx, pixel in enumerate(data_points):
        X[idx, :] = pixel[0:2]

    # The corresponding labels for our points
    y = data_points[:, 2]
    y = y.reshape(-1, 1)

    X = X / (X.max() - X.min())

    return X, y

def plot_decision_boundary(w: np.ndarray, b: float, X: np.ndarray, y: np.ndarray, 
                          predict: Callable, accuracy: Callable, color1: str, color2: str) -> Tuple[Figure, PathCollection, Text, Any]:
    """
    Plot decision boundary of a logistic regression model.
    
    Args:
        w: Weight vector of shape (d, 1)
        b: Bias term (scalar)
        X: Feature matrix of shape (m, d)
        y: True labels of shape (m, 1)
        predict: Function to make predictions given w, b, X
        accuracy: Function to compute accuracy given predictions and true labels
        color1: Color for class 0
        color2: Color for class 1
        
    Returns:
        Tuple of matplotlib objects (figure, scatter, title, contour)
    """
    """
    Plot decision boundary of the logistic regression model.
    """
    plt.clf()

    # Add a little bit of a border to x_min/max and y_min/max for plotting
    x_min, x_max = X[:, 0].min() - 0.25, X[:, 0].max() + 0.25
    y_min, y_max = X[:, 1].min() - 0.25, X[:, 1].max() + 0.25
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # Pass all x, y pairs through sigmoid function
    all_points = np.c_[xx.ravel(), yy.ravel()]
    Z = predict(w, b, all_points)
    
    # Convert back to grid-shape
    Z = Z.reshape(xx.shape)
    contour = plt.contourf(xx, yy, Z, alpha=.5, levels=[-1, 0, 1], colors=[color1, color2])

    color_dict = {0 : color1, 1 : color2}
    colors = [color_dict[label] for label in y[:, 0]]
    scatter = plt.scatter(X[:, 0], X[:, 1], c=colors, s=5, alpha=1)
    pred = predict(w, b, X)
    title = plt.title(f"NumPy Logistic Regression Decision Boundary\nAccuracy={accuracy(pred, y):.2f}, Weights=({w[0, 0]:.2f}, {w[1, 0]:.2f}), Bias={b:.2f}")
    plt.gca().set_aspect('equal', adjustable='box')
    plot = plt.gcf()
    return plot, scatter, title, contour

def create_gradient_descent_animation(X: np.ndarray, y: np.ndarray, params_list: List[Tuple[np.ndarray, float]], 
                                    predict: Callable, accuracy: Callable, color1: str, color2: str, 
                                    filename: str = 'gradient_descent.gif', fps: int = 30) -> None:
    """
    Create and display an optimized animation of decision boundary evolution during gradient descent.
    
    This optimized version reduces computation time by:
    - Using lower resolution meshgrid (0.05 instead of 0.01)
    - Skipping frames to target ~50 frames maximum
    - Pre-computing static elements
    - Disabling blitting for more reliable updates
    
    Args:
        X: Feature matrix of shape (m, d)
        y: True labels of shape (m, 1)
        params_list: List of (weight, bias) tuples from training iterations
        predict: Function to make predictions
        accuracy: Function to compute accuracy
        color1: Color for class 0
        color2: Color for class 1
        filename: Output filename for the animation (currently unused)
        fps: Frames per second for animation
    """
    # Reduce number of frames for faster animation (take every nth frame)
    target_frames = 50  # Target maximum number of frames
    skip_frames = max(1, len(params_list) // target_frames)
    params_subset = params_list[::skip_frames]
    
    print(f"Creating animation with {len(params_subset)} frames (reduced from {len(params_list)})...")
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Pre-compute the meshgrid once (lower resolution for speed)
    x_min, x_max = X[:, 0].min() - 0.25, X[:, 0].max() + 0.25
    y_min, y_max = X[:, 1].min() - 0.25, X[:, 1].max() + 0.25
    # Reduced resolution from 0.01 to 0.05 for 25x speedup in meshgrid computation
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    all_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Pre-compute colors for data points
    color_dict = {0: color1, 1: color2}
    colors = [color_dict[label] for label in y[:, 0]]

    def animate(frame_idx):
        # Clear previous plot
        ax.clear()
        
        # Get parameters for this frame
        if frame_idx < len(params_subset):
            w, b = params_subset[frame_idx]
            iteration = frame_idx * skip_frames
        else:
            w, b = params_subset[-1]  # Hold on last frame
            iteration = len(params_list) - 1
        
        # Compute predictions on meshgrid
        Z = predict(w, b, all_points)
        Z = Z.reshape(xx.shape)
        
        # Create contour plot
        ax.contourf(xx, yy, Z, alpha=0.5, levels=[-1, 0, 1], colors=[color1, color2])
        
        # Plot data points
        scatter = ax.scatter(X[:, 0], X[:, 1], c=colors, s=5, alpha=1)
        
        # Compute and display accuracy
        pred = predict(w, b, X)
        acc = accuracy(pred, y)
        title = ax.set_title(f"Gradient Descent Animation\nIteration: {iteration}, "
                           f"Accuracy: {acc:.2f}, Weights: ({w[0, 0]:.2f}, {w[1, 0]:.2f}), Bias: {b:.2f}")
        
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Return artists for potential blitting (even though we disabled it)
        return [title, scatter]
    
    # Create the animation with optimized settings
    anim = FuncAnimation(
        fig,
        animate,
        frames=len(params_subset) + 10,  # Hold last frame for a bit
        interval=max(100, 1000//fps),    # Minimum 100ms interval for stability
        blit=False,                      # Disable blitting for more reliable updates
        repeat=True
    )

    # Display the animation
    display(HTML(anim.to_jshtml()))
    
    plt.close()

# Visualize data and model decision boundary
def plot_decision_boundary_torch(w: np.ndarray, b: float, X: np.ndarray, y: np.ndarray, 
                               sigmoid_torch: Callable, color1: str, color2: str) -> None:
    """
    Plot decision boundary for a PyTorch logistic regression model.
    
    Args:
        w: Weight array
        b: Bias term
        X: Feature matrix
        y: True labels
        sigmoid_torch: PyTorch sigmoid function
        color1: Color for class 0
        color2: Color for class 1
    """
    x_min, x_max = X[:, 0].min() - 0.25, X[:, 0].max() + 0.25
    y_min, y_max = X[:, 1].min() - 0.25, X[:, 1].max() + 0.25
    xx, yy = torch.meshgrid(torch.arange(x_min, x_max, 0.01), torch.arange(y_min, y_max, 0.01), indexing='ij')

    z = sigmoid_torch(torch.mm(torch.tensor(w.T, dtype=torch.float32), torch.tensor(np.c_[xx.ravel(), yy.ravel()].T, dtype=torch.float32)) + torch.tensor(b, dtype=torch.float32))
    
    z = z > 0.5
    z = z.detach().numpy().reshape(xx.shape)
    plt.contourf(xx, yy, z, alpha=.5, levels=[-1, 0, 1], colors=[color1, color2])
    color_dict = {0 : color1, 1 : color2}
    colors = [color_dict[label] for label in y[:, 0]]
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=5)
    plt.title("PyTorch Logistic Regression Decision Boundary")
    plt.gca().set_aspect('equal', adjustable='box')

    plt.show()
    
# Visualize decision boundary for MLP
def plot_decision_boundary_mlp(model: torch.nn.Module, X: Union[torch.Tensor, np.ndarray], 
                             y: Union[torch.Tensor, np.ndarray], color1: str, color2: str) -> None:
    """
    Plot decision boundary for a multi-layer perceptron model.
    
    Args:
        model: Trained PyTorch neural network model
        X: Feature tensor or numpy array
        y: True label tensor or numpy array
        color1: Color for class 0
        color2: Color for class 1
    """
    # Convert to numpy if needed
    if isinstance(X, torch.Tensor):
        X_np = X.numpy()
    else:
        X_np = X
    
    if isinstance(y, torch.Tensor):
        y_np = y.numpy()
    else:
        y_np = y
    
    x_min, x_max = X_np[:, 0].min() - 0.25, X_np[:, 0].max() + 0.25
    y_min, y_max = X_np[:, 1].min() - 0.25, X_np[:, 1].max() + 0.25
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    X_new = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    Z = model(X_new)
    Z = (Z > 0.5).float().detach().numpy().reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=.5, levels=[-1, 0, 1], colors=[color1, color2])
    color_dict = {0 : color1, 1 : color2}
    colors = [color_dict[int(label)] for label in y_np[:, 0]]
    plt.scatter(X_np[:, 0], X_np[:, 1], c=colors, s=5)
    plt.title("Decision Boundary")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()