# Desirability Profile Functions from Derringer and Suich (1980)
# These functions help optimize multiple response variables simultaneously

import numpy as np

def one_sided_desirability(y, y_min, y_max, r=1):
    """
    Desirability function for cases where larger values are better.
    
    Parameters:
    y (float): Observed value
    y_min (float): Minimum acceptable value
    y_max (float): Maximum/ideal value 
    r (float): Shape parameter (default=1 for linear)
    
    Returns:
    float: Desirability score between 0 and 1
    """
    if y < y_min:
        return 0
    elif y > y_max:
        return 1
    else:
        return ((y - y_min)/(y_max - y_min))**r

def negative_one_sided_desirability(y, y_min, y_max, r=1):
    """
    Desirability function for cases where smaller values are better.
    
    Parameters:
    y (float): Observed value
    y_min (float): Minimum/ideal value
    y_max (float): Maximum acceptable value
    r (float): Shape parameter (default=1 for linear)
    
    Returns:
    float: Desirability score between 0 and 1
    """
    if y < y_min:
        return 1
    elif y > y_max:
        return 0
    else:
        return ((y_max - y)/(y_max - y_min))**r

def two_sided_desirability(y, y_min, y_max, y_target, r1=1, r2=1):
    """
    Desirability function for cases where a target value is optimal.
    
    Parameters:
    y (float): Observed value
    y_min (float): Minimum acceptable value
    y_max (float): Maximum acceptable value
    y_target (float): Target/optimal value
    r1, r2 (float): Shape parameters for left and right sides (default=1 for linear)
    
    Returns:
    float: Desirability score between 0 and 1
    """
    if y < y_min or y > y_max:
        return 0
    elif y <= y_target:
        return ((y - y_min)/(y_target - y_min))**r1
    else:
        return ((y_max - y)/(y_max - y_target))**r2

def geometric_mean_desirability(desirabilities, weights=None):
    """
    Calculate the overall desirability using geometric mean.
    
    Parameters:
    desirabilities (list): List of individual desirability scores
    weights (list): List of weights for each desirability (default=None for equal weights)
    
    Returns:
    float: Overall desirability score between 0 and 1
    """
    import numpy as np
    
    if weights is None:
        weights = [1] * len(desirabilities)
    
    weights_sum = sum(weights)
    weights = [w/weights_sum for w in weights]  # Normalize weights
    
    return np.prod([d**w for d, w in zip(desirabilities, weights)])

def segmented_one_sided_desirability(y, y_min, y_max, y_mid, gamma=0.5, r1=1, r2=1):
    """
    Segmented desirability function with a middle point that divides the range into two segments.
    This allows for different scaling in different regions of the response range.
    Based on the work of In-Jun Jeong and Kwang-Jae Kim (2006).
    
    Parameters:
    y (float): Observed value
    y_min (float): Minimum acceptable value
    y_max (float): Maximum acceptable value
    y_mid (float): Middle point that divides the range into two segments
    r1 (float): Shape parameter for first segment (default=1 for linear)
    r2 (float): Shape parameter for second segment (default=1 for linear)
    
    Returns:
    float: Desirability score between 0 and 1
    """
    if y < y_min:
        return 0
    elif y > y_max:
        return 1
    elif y <= y_mid:
        # First segment scaling (y_min to y_mid)
        return gamma * ((y - y_min) / (y_mid - y_min))**r1
    else:
        # Second segment scaling (y_mid to y_max)
        return (1-gamma) + (1-gamma) * ((y - y_mid) / (y_max - y_mid))**r2

def segmented_negative_one_sided_desirability(d, d_min, d_max, d_mid, **kwargs):
    """
    Segmented desirability function for cases where smaller values are better.
    
    Parameters:
    d (float): Observed value
    d_min (float): Minimum/ideal value
    d_max (float): Maximum acceptable value
    d_mid (float): Middle point that divides the range
    
    Optional kwargs:
    gamma (float): Baseline desirability at d_mid (default=0.5)
    r1 (float): Shape parameter for first segment (default=1)
    r2 (float): Shape parameter for second segment (default=1)
    
    Returns:
    float: Desirability score between 0 and 1
    """
    # Extract optional parameters with defaults
    gamma = kwargs.get('gamma', 0.5)
    r1 = kwargs.get('r1', 1)
    r2 = kwargs.get('r2', 1)

    if d < d_min:
        return 1
    elif d > d_max:
        return 0
    elif d <= d_mid:
        # First segment scaling (d_min to d_mid)
        return (gamma) + (1-gamma) * (1-(d - d_min) / (d_mid - d_min))**r1
    else:
        # Second segment scaling (d_mid to d_max)
        return gamma - gamma * ((d - d_mid) / (d_max - d_mid))**r2
