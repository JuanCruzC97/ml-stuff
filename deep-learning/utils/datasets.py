from this import d
import pandas as pd
from sklearn.datasets import make_s_curve

def make_dataset(n_samples, noise, random_state):
    
    X, y = make_s_curve(n_samples=n_samples, noise=noise, random_state=random_state)
    
    df = pd.DataFrame(X)
    df.columns = ["X1", "X2", "X3"]
    df["y"] = y
    
    return df