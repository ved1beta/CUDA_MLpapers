import numpy as np
from typing import List, Callable

def local_sgd_synchronous(
    X: np.ndarray,
    y: np.ndarray,
    loss_gradient: Callable,
    K: int,
    H: int,
    T: int,
    eta: float,
    theta_init: np.ndarray
) -> np.ndarray:
    """ K: Number of workers
        H: Number of local steps before synchronization
        T: Total number of iterations"""

    n = len(X)
    
    
    theta = [theta_init.copy() for _ in range(K)]
    
    for t in range(T):
        for k in range(K):
            i_k = np.random.randint(0, n)
            
            if (t + 1) % H == 0:
  
                theta_avg = np.mean([theta[j] for j in range(K)], axis=0)
                

                grad = loss_gradient(theta_avg, X[i_k], y[i_k])
                theta[k] = theta_avg - eta * grad
            else:
                grad = loss_gradient(theta[k], X[i_k], y[i_k])
                theta[k] = theta[k] - eta * grad
    return np.mean(theta, axis=0)

def local_sgd_asynchronous(
    X: np.ndarray,
    y: np.ndarray,
    loss_gradient: Callable,
    K: int,
    H: int,
    T: int,
    eta: float,
    theta_init: np.ndarray
) -> np.ndarray:
    """
        K: Number of workers
        H: Number of local steps before synchronization
        T: Total number of iterations

    """
    n = len(X)
    

    theta = [theta_init.copy() for _ in range(K)]
    r = [0 for _ in range(K)]
    
    theta_bar = theta_init.copy()
    
    for t in range(T):
        for k in range(K):
 
            i_k = np.random.randint(0, n)

            grad = loss_gradient(theta[k], X[i_k], y[i_k])
            theta[k] = theta[k] - eta * grad
            
            # Check if it's time to synchronize
            if (t + 1) % H == 0:
                # Atomic aggregation: add this worker's update to global aggregate
                # add(theta_bar, (1/K) * (theta[k] - theta_old[k]))
                # Here we simplify by directly updating the aggregate
                delta = theta[k] - theta_bar
                theta_bar = theta_bar + (1.0 / K) * delta
                

                theta[k] = theta_bar.copy()

                r[k] = t + 1
    
    return theta_bar