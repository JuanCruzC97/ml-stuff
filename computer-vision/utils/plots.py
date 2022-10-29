# Permite hacer el plot interactivo de imágenes pasando un array.
import numpy as np
import plotly.express as px
from predictions import prediction_to_class
from operations import transform_img

def plot_img(img_array, targets=None, grayscale=False):

  color_scale = "gray" if grayscale else "viridis"
  shape = img_array.shape

  if shape[:-1] == 1:
    img_array = img_array.reshape(shape[:-1])

  plot = px.imshow(img_array,
                   color_continuous_scale=color_scale,
                   height=700,
                   width=600,
                   template="simple_white")
  
  plot.update_layout(xaxis={"showticklabels":False},
                     yaxis={"showticklabels":False},
                     coloraxis={"colorbar":{"orientation":"h",
                                            "thickness":15,
                                            "y":-0.05,
                                            "yanchor":"bottom"}})
  if targets is not None:
    target_name = prediction_to_class(targets[0])

    plot.update_layout(title={"text":f'Label = {target_name}',
                              "font":{"size":20},
                              "x":0.5,
                              "y":0.9,
                              "xanchor":"center"})
  
  return plot



# Gráfico de métricas durante el entrenamiento.
def plot_train_metrics(data, metric_train_list, metric_val_list=None, title=None):

  data["epochs"] = list(data.index)
  data = data.melt(id_vars = "epochs", var_name="metrics", value_name="value")

  if metric_val_list == None:
    for metric in metric_train_list:

      data_plot = data.query("metrics == @metric")

      plot = px.line(data_frame=data_plot,
                     x="epochs",
                     y="value",
                     height=500,
                     width=1000,
                     color_discrete_sequence=["#035397"],
                     title=title,
                     template="plotly_white")
      
      plot.update_layout(xaxis={"title":{"text":"Epochs"}},
                         yaxis={"title":{"text":metric.capitalize()}},
                         hovermode="x unified",
                         legend={"title":{"text":""},
                                 "orientation":"h",
                                 "yanchor":"bottom",
                                 "y":1.02,
                                 "xanchor":"right",
                                 "x":1})

      plot.show()
  
  else:
    for metric_train, metric_val in zip(metric_train_list, metric_val_list):

      data_plot = data.query("metrics in [@metric_train, @metric_val]")

      plot = px.line(data_frame=data_plot,
                     x="epochs",
                     y="value",
                     color="metrics",
                     height=500,
                     width=1000,
                     color_discrete_sequence=["#035397", "#E8630A"],
                     #custom_data=["epochs", metric_train, metric_val],
                     title=title,
                     template="plotly_white")
      
      plot.update_layout(xaxis={"title":{"text":"Epochs"}},
                         yaxis={"title":{"text":metric_train.capitalize()}},
                         hovermode="x unified",
                         legend={"title":{"text":""},
                                 "orientation":"h",
                                 "yanchor":"bottom",
                                 "y":1.02,
                                 "xanchor":"right",
                                 "x":1})
      
      plot.update_traces(hovertemplate=None)

      plot.show()
      
      
      
def plot_transformed_img(img_array, filter, weight=1, pad=True):

  image, conv, max_pool = transform_img(img_array, filter, weight, pad=pad)
  plots = [image, conv, max_pool]
  names = ["image", "conv", "max_pool"]

  data_shape = list(image.shape)
  data_shape.insert(0, 3)

  data = np.zeros(data_shape)

  for i, plot in enumerate([image, conv, max_pool]):
    data[i] = plot
  
  plot = px.imshow(data, facet_col=0)

  for i, name in enumerate(names):
    plot.layout.annotations[i]['text'] = f'{name}'

  return plot