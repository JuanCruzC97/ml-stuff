import pandas as pd

def prediction_to_class(x):
  
  labels = {0:"Forests",
            1:"Mountains",
            2:"Sea",
            3:"Street"}

  return labels[x]



def model_evaluation(model, X_train, y_train, X_val, y_val, X_test, y_test, result=False):
  eval = dict(train = model.evaluate(X_train, y_train, verbose=0, return_dict=True, batch_size=32),
              val = model.evaluate(X_val, y_val, verbose=0, return_dict=True, batch_size=32),
              test = model.evaluate(X_test, y_test, verbose=0, return_dict=True, batch_size=32))

  df_eval = pd.DataFrame(eval)
  display(df_eval)

  if result:
    return df_eval