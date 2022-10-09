import pandas as pd
from sklearn.metrics import confusion_matrix

def display_confusion_matrix(y_true, y_pred):
    
    m = pd.DataFrame(confusion_matrix(y_true=y_true, y_pred=y_pred), 
                     index=["0 True", "1 True"], 
                     columns=["0 Pred", "1 Pred"])
    return m