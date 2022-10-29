import numpy as np
import tensorflow as tf
import cv2
import os

# Usando una lista de nombres de imágenes en un directorio genera un dataset con imágenes y labels.
def create_img_dataset(directory, img_list, labels_list, img_shape, scale=False, color_mode="rgb"):
  
  samples = len(img_list)
  shape = list(img_shape)
  shape.insert(0, samples)

  data = np.zeros(shape)
  targets = np.zeros((samples,1))

  for i, img in enumerate(img_list):

    try:
      data[i] = import_img_as_array(directory, img, color_mode)

    except:
      img_t = import_img_as_array(directory, img, color_mode)
      img_t = cv2.resize(img_t, dsize=tuple(shape[1:3]), interpolation=cv2.INTER_LINEAR)
      data[i] = img_t


    if scale:
      data[i] = data[i]/255

    targets[i] = labels_list[i]
  
  #targets = targets.ravel()

  return data, targets


# Importa una imágen de un directorio como un array.
def import_img_as_array(directory, img_path, color_mode="rgb"):

  path = os.path.join(directory, img_path)
  img = tf.keras.preprocessing.image.load_img(path, color_mode=color_mode)
  array = tf.keras.preprocessing.image.img_to_array(img)

  return array