#--------------------------------------------------------
#
#  Stable Diffusion Tutorial
#  from MoniGarr.com
#  Tutorial Link: 
#	https://medium.com/@monigarr_3883/how-to-install-and-run-stable-diffusion-on-your-windows-pc-ca43e6beb150
# 
#--------------------------------------------------------

#-----------------------------------------------------
# imports
#-----------------------------------------------------
import time
import keras_cv
from tensorflow import keras
import matplotlib.pyplot as plt

model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)

#------------------------------------------------------
# Prompt String 
# "Forest landsape with distant river"
# Change prompt string to anything you imagine.
#------------------------------------------------------
images = model.text_to_image("Forest landscape with distant river", batch_size=1)

def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")

plot_images(images)

plt.show()