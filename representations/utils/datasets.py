import numpy as np
import pandas as pd


def make_classification_dataset(n_samples, noise=0.1, factor_0=0.5, factor_1=0.2, random_state=1):
    """Esta función genera un dataset de clasificación binaria donde cada clase tiene
    forma circular. Se puede definir el número de observaciones, el nivel de ruido y 
    los radios de ambas clases con los parámetros factor.

    Args:
        n_samples (int): Número de observaciones totales del set de datos generado.
        noise (float, optional): Nivel de ruido en los datos. Defaults to 0.1.
        factor_0 (float, optional): Factor que define la magnitud de los valores de la clase 0. Defaults to 0.5.
        factor_1 (float, optional): Factor que define la magnitud de los valores de la clase 1. Defaults to 0.2.
        random_state (int, optional): Semilla para números aleatorios. Defaults to 1.

    Returns:
        pd.DataFrame: Set de datos con 2 columnas de variables explicativas 'X1' y 'X2'
        y una variable respuesta binaria 'y'.
    """
    
    # Cada clase tendrá la mitad del número de observaciones definidos.
    # En caso de elegir un número impar tendremos un número de observaciones mayor que n_samples.
    n_samples_per_class = int(n_samples/2) 
    
    # Generamos los valores para cada clase.
    np.random.seed(random_state)
    samples_0 = np.random.uniform(0, 2 * np.pi, size=n_samples_per_class)
    samples_1 = np.random.uniform(0, 2 * np.pi, size=n_samples_per_class)
    
    # Generación de los valores en x e y de cada punto de cada clase transformados.
    x1_0 = np.cos(samples_0) * factor_0
    x2_0 = np.sin(samples_0) * factor_0
    x1_1 = np.cos(samples_1) * factor_1
    x2_1 = np.sin(samples_1) * factor_1
    
    # Creamos las columnas y agregamos el ruido aleatorio.
    x1 = np.append(x1_0, x1_1)
    x1 += np.random.normal(0, noise, n_samples_per_class*2)
    
    x2 = np.append(x2_0, x2_1)
    x2 += np.random.normal(0, noise, n_samples_per_class*2)
    
    # Generamos la variable respuesta binaria.
    target = np.append(np.zeros(n_samples_per_class, dtype=int), np.ones(n_samples_per_class, dtype=int))
    
    # Creamos el dataset de output.
    df = pd.DataFrame({"X1":x1,
                       "X2":x2,
                       "y":target})
    
    # Definimos a la variable respuesta como variable categórica.
    # Mezcla las observaciones.
    df = (df
          .astype({"y":"category"})
          .sample(frac=1)
          .reset_index(drop=True))
    
    return df