import torch
import numpy as np

def calculate_asymmetric_zero_point(min_val, max_val, qmin=-128, qmax=127):

    min_val = min(0.0, min_val)
    max_val = max(0.0, max_val)
    
    scale = (max_val - min_val) / (qmax - qmin)
    
    zero_point = qmin - (min_val / scale)
    
    zero_point = int(round(zero_point))
    zero_point = max(qmin, min(qmax, zero_point))
    
    return scale, zero_point

