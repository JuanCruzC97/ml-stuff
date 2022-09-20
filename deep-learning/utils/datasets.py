from this import d
import pandas as pd
import numpy as np
from sklearn.datasets import make_s_curve

def make_dataset(n_samples, noise, random_state):
    
    X, y = make_s_curve(n_samples=n_samples, noise=noise, random_state=random_state)
    Z = np.hstack((X,y.reshape(n_samples,1)))
    
    df = pd.DataFrame(Z)
    df.columns = ["y", "X2", "X3", "X1"]
    
    df = (df
          [["X1", "X2", "X3", "y"]]
          .assign(y=df.y + df.X1*0.3 + 0.1))
    
    
    return df