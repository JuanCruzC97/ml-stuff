import numpy as np

def apply_convolution(img_array, filter, weight=1):
  # Copy image to a numpy array
  image_transformed = np.copy(img_array)

  # Get the dimensions of the image
  size_x = image_transformed.shape[0]
  size_y = image_transformed.shape[1]

  # Iterate over the image
  for x in range(1,size_x-1):
    for y in range(1,size_y-1):
        convolution = 0.0
        convolution = convolution + (img_array[x-1, y-1] * filter[0][0])
        convolution = convolution + (img_array[x-1, y] * filter[0][1])  
        convolution = convolution + (img_array[x-1, y+1] * filter[0][2])     
        convolution = convolution + (img_array[x, y-1] * filter[1][0])    
        convolution = convolution + (img_array[x, y] * filter[1][1])    
        convolution = convolution + (img_array[x, y+1] * filter[1][2])    
        convolution = convolution + (img_array[x+1, y-1] * filter[2][0])    
        convolution = convolution + (img_array[x+1, y] * filter[2][1])    
        convolution = convolution + (img_array[x+1, y+1] * filter[2][2])    
        
        # Multiply by weight
        convolution = convolution * weight   
        
        # Check the boundaries of the pixel values
        if(convolution<0):
          convolution=0
        if(convolution>1):
          convolution=1

        # Load into the transformed image
        image_transformed[x, y] = convolution

  return image_transformed


def apply_max_pooling(img_array, pad=True):

  # Copy image to a numpy array
  image_transformed = np.copy(img_array)

  # Get the dimensions of the image
  size_x = image_transformed.shape[0]
  size_y = image_transformed.shape[1]

  # Assign dimensions half the size of the original image
  new_x = int(size_x/2)
  new_y = int(size_y/2)

  # Create blank image with reduced dimensions
  new_image = np.zeros((new_x, new_y))

  # Iterate over the image
  for x in range(0, size_x, 2):
    for y in range(0, size_y, 2):
      
      # Store all the pixel values in the (2,2) pool
      pixels = []
      pixels.append(image_transformed[x, y])
      pixels.append(image_transformed[x+1, y])
      pixels.append(image_transformed[x, y+1])
      pixels.append(image_transformed[x+1, y+1])

      # Get only the largest value and assign to the reduced image
      new_image[int(x/2),int(y/2)] = max(pixels)
  
  if pad:
    padding = np.zeros((size_x, size_y))
    x_min = int(size_x/4)
    x_max = int((size_x/4) * 3)
    y_min = int(size_y/4)
    y_max = int((size_y/4) * 3)
    padding[x_min:x_max, y_min:y_max] = new_image

    return padding

  else:
    return new_image


def transform_img(img_array, filter, weight=1, pad=True):
  
  image = np.copy(img_array)

  if len(image.shape) == 3:
    image = image.reshape(image.shape[:-1])

  conv = apply_convolution(image, filter, weight)
  max_pool = apply_max_pooling(conv, pad=pad)

  return (image, conv, max_pool)