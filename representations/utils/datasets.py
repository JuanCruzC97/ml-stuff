import numpy as np
import pandas as pd


def make_classification_dataset(n_samples, noise, inner_factor, outer_factor, random_state):
    
    n_samples_per_class = int(n_samples/2) 
   
    np.random.seed(random_state)
    samples_in = np.random.uniform(0, 2 * np.pi, size=n_samples_per_class)
    samples_out = np.random.uniform(0, 2 * np.pi, size=n_samples_per_class)
    
    x_in = np.cos(samples_in) * inner_factor
    y_in = np.sin(samples_in) * inner_factor
    x_out = np.cos(samples_out) * outer_factor
    y_out = np.sin(samples_out) * outer_factor
    
    x1 = np.append(x_in, x_out)
    x1 += np.random.normal(0, noise, n_samples_per_class*2)
    
    x2 = np.append(y_in, y_out)
    x2 += np.random.normal(0, noise, n_samples_per_class*2)
    
    target = np.append(np.zeros(n_samples_per_class, dtype=int), np.ones(n_samples_per_class, dtype=int))
    
    df = pd.DataFrame({"X1":x1,
                       "X2":x2,
                       "y":target})
    
    df = df.astype({"y":"category"})
    
    return df