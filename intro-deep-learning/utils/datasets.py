from this import d
import pandas as pd
import numpy as np
from tensorflow import keras

def make_regression_dataset(n_samples, noise, random_state):
    
    np.random.seed(random_state)
    x = np.random.uniform(-4, 4, size=n_samples)
    y = 0.95*x + 2.5*np.sin(x) + 0.5 + 0.25*x**2 + np.random.normal(0, noise, n_samples)
    df = pd.DataFrame({"X":x, "y":y})
    
    return df


def get_fit_data(history, loss_name='loss'):
    
    df = pd.DataFrame(history.history)
    
    df = (df
          .set_index(np.arange(1, df.shape[0]+1))
          .rename(columns={'loss':loss_name}))
    
    return df


def get_training_preds(data, model, batch_size, epochs):
    
    # Hay que definir antes el modelo nuevo y pasarlo..
    
    df_weights = pd.DataFrame(columns=["weight", "bias"], index=range(EPOCHS))
    df_weights.loc[0, "weight"] = model.get_weights()[0][0][0]
    df_weights.loc[0, "bias"] = model.get_weights()[1][0]
    
    df_preds = data[["X", "y"]].copy()
    df_preds["y_0"] = model.predict(df_preds["X"], verbose=0)

    for epoch in range(1, epochs+1):
        
        model.fit(x=data[["X"]], y=data["y"], batch_size=batch_size, epochs=1, shuffle=True, verbose=0)
        
        df_weights.loc[epoch, "weight"] = model.get_weights()[0][0][0]
        df_weights.loc[epoch, "bias"] = model.get_weights()[1][0]
        
        df_preds[f'y_{epoch}'] = model.predict(df_preds["X"], verbose=0)
        
        
    
    df_preds = pd.melt(df_preds, 
                       id_vars=["X", "y"], 
                       value_vars=[f'y_{epoch}' for epoch in range(0, epochs+1)],
                       var_name="epoch",
                       value_name="y_pred")

    df_preds = (df_preds
                .assign(epoch=lambda df: df.epoch.str.replace("y_", ""))
                .astype({"epoch":"int"}))

    return (df_weights, df_preds)