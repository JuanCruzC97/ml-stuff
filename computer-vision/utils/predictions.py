import pandas as pd
from sklearn.metrics import confusion_matrix

def prediction_to_class(x):
  
  labels = {0:"Forests",
            1:"Mountains",
            2:"Sea",
            3:"Street"}

  return labels[x]



def model_evaluation(model, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None, result=False):
    
    #train_eval = model.evaluate(X_train, y_train, verbose=0, return_dict=True, batch_size=32)
    
    eval = dict(train = model.evaluate(X_train, y_train, verbose=0, return_dict=True, batch_size=32),
                val = model.evaluate(X_val, y_val, verbose=0, return_dict=True, batch_size=32),
                test = model.evaluate(X_test, y_test, verbose=0, return_dict=True, batch_size=32))

    df_eval = pd.DataFrame(eval)
    display(df_eval)

    if result:
        return df_eval


def display_confusion_matrix(y_true, y_pred):
    # No funciona si el y_true pasado no incluye al menos una observacion de cada clase.
    
    m = pd.DataFrame(confusion_matrix(y_true=y_true, y_pred=y_pred)) 

    labels = ["Forests", "Mountains", "Sea", "Street"]

    m.index = pd.MultiIndex.from_tuples([("True", label) for label in labels])
    m.columns = pd.MultiIndex.from_tuples([("Pred", label) for label in labels])
    
    display(m)
