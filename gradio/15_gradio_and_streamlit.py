#!/usr/bin/env python
# coding: utf-8

# #### https://github.com/HVL-ML/DAT255/blob/main/notebooks/15_gradio_and_streamlit.ipynb er brukt som utgangspunkt i denne notebooken.

# # Gradio (and Streamlit) deployment
# 
# For deploying an ML-based app there are various approaches, but [Gradio](gradio.app) and [Streamlit](https://streamlit.io/) are both quick and convenient ways to do so. Typically we would implement this in a plain python (`.py`) file rather than a `.ipynb` notebook, but Gradio works here too, so let's try that first. Streamlit needs to be run directly in python, but the code is given below, so you can try out that too.
# 
# In this example we set up an image classifier, where the user can upload an image and get the top 5 class predictions in return.

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)
from PIL import Image


# ## Set up a pretrained model
# 
# Let's download and set up a `MobileNetV2` model, trained on the 1000 classes in the ImageNet dataset. You can change this to anything you like. Remeber, however, to also use the appropriate preprocessing function.

# In[3]:


# model = MobileNetV2(weights="imagenet")

# The model loading below is from Deep Learning with Python, Third edition (https://deeplearningwithpython.io). Chapter: 8, Fitting the model
model1 = keras.models.load_model(
    "train_hierarchical_cnn_global_aug_v2.keras"
)
# The model loading below is from Deep Learning with Python, Third edition (https://deeplearningwithpython.io). Chapter: 8, Fitting the model
model2 = keras.models.load_model(
    "hierarchical_cnn_global_final_v2.keras"
)
# The model loading below is from Deep Learning with Python, Third edition (https://deeplearningwithpython.io). Chapter: 8, Fitting the model
model3 = keras.models.load_model(
    "tl2_efficientnetv2s_finetune60.keras"
)

IMG_RES = (300, 300)


# # Cellene under kommer tilnærmet direkte fra et annet prosjekt gruppen har arbeidet med i faget, DAT255. Referanse: A. Kidess, J. O. Rosberg, og J. Kaste, «Upublisert egen semesteroppgave i faget, DAT255: Explainability methods for image classification», Upublisert egen semesteroppgave i faget, DAT255, Høgskulen på Vestlandet, Haugesund, 2026.

# #### The two cells below are based on Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps, Image-Specific Class Saliency Visualisation (https://arxiv.org/abs/1312.6034)

# In[4]:


# This method is based on Deep Learning with Python, Third edition (https://deeplearningwithpython.io). Chapter: 2, Gradient computation,
# and get_gradients from https://keras.io/examples/vision/integrated_gradients/
def get_gradients(img_array, lvl, model, lvl2_pred_index):
    with tf.GradientTape() as tape:
        tape.watch(img_array)
        # print(img_array)
        if lvl2_pred_index != None:
            score = model(img_array)[lvl][0][lvl2_pred_index]
        else:
            # Denne veiledningen på reduce_max ble brukt: https://www.tensorflow.org/api_docs/python/tf/math/reduce_max
            score = tf.math.reduce_max(model(img_array)[lvl])
    gradients = tape.gradient(score, img_array)
    return gradients


# In[5]:


def get_heatmap(img_array, lvl, model, lvl2_pred_index=None):
    gradients = get_gradients(img_array, lvl, model, lvl2_pred_index)
    abs_gradients = np.abs(gradients[0])
    # np.max is from: Deep Learning with Python, Third edition (https://deeplearningwithpython.io). Chapter: 10, Displaying the class activation heatmap
    max_gradient = np.max(abs_gradients)

    if max_gradient == 0.0:
        max_gradient = 0.000000000001

    pixel_gradients = ((abs_gradients / max_gradient) * 255.0)


    # Denne dokumentasjonen er brukt for å lage linjen under: https://numpy.org/doc/2.2/reference/generated/numpy.max.html
    return np.max(pixel_gradients, 2)


# #### The idea of the method below is from: SmoothGrad: removing noise by adding noise, Capping outlying values (https://arxiv.org/pdf/1706.03825)

# In[6]:


import math

def cap_heatmap(heatmap):
    values = []
    for i in range(len(heatmap)):
        for j in range(len(heatmap[i])):
            values.append(heatmap[i][j])
    values.sort()

    ninty_nine_percentile = values[math.floor(len(values) * 0.995)]

    for i in range(len(heatmap)):
        for j in range(len(heatmap[i])):
            if (heatmap[i][j] > ninty_nine_percentile):
                heatmap[i][j] = ninty_nine_percentile

    # np.max is from: Deep Learning with Python, Third edition (https://deeplearningwithpython.io). Chapter: 10, Displaying the class activation heatmap
    max_value = np.max(heatmap)
    if (max_value == 0.0):
        max_value = 0.000000000001

    heatmap = (heatmap / max_value) * 255.0
    return heatmap


# ### The below cell is strongly based on Deep Learning with Python, Third edition (https://deeplearningwithpython.io). Chapter: 10, Visualizing heatmaps of class activation

# In[7]:


import matplotlib.cm as cm

def superimpose(img_array, heatmap):
    # Uses the "jet" colormap to recolorize the heatmap
    jet = cm.get_cmap("jet")

    jet_colors = jet(np.arange(256))[:, :3]
    jet_colors-=[0,0,0.5]

    # Convertion to int is from: https://www.w3schools.com/python/numpy/numpy_data_types.asp (Converting Data Type on Existing Arrays)
    jet_heatmap = jet_colors[(np.round(heatmap)).astype('i')]

    # Superimposes the heatmap and the original image
    superimposed_img = jet_heatmap / 3.0 + img_array / 3.0
    superimposed_img = keras.utils.array_to_img(superimposed_img)
    return superimposed_img


# # Cellene over kommer tilnærmet direkte fra et annet prosjekt gruppen har arbeidet med i faget, DAT255. Referanse: A. Kidess, J. O. Rosberg, og J. Kaste, «Upublisert egen semesteroppgave i faget, DAT255: Explainability methods for image classification», Upublisert egen semesteroppgave i faget, DAT255, Høgskulen på Vestlandet, Haugesund, 2026.

# In[8]:


# import imageio
import gradio as gr

from preprocessing import preprocess_image_tensor

def classify_image(img: Image.Image, checkbox_group, sg_checkbox):

    # Resize to the input image to what MobileNet expects
    # img_resized = img.resize(IMG_RES)
    arr = np.array(img)

    # Check the color channels, so we can take both grayscale, RGB, and RGBA images as input.
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    # elif arr.shape[-1] == 4:
    #     arr = arr[..., :3]

    arr = arr / 255.0

    print(arr)

    # Linjen under er basert på: https://www.tensorflow.org/api_docs/python/tf/keras/ops/convert_to_tensor
    # arr = keras.ops.convert_to_tensor(arr, dtype="float32")

    arr = preprocess_image_tensor(arr, IMG_RES[0])
    # np.max is from: Deep Learning with Python, Third edition (https://deeplearningwithpython.io). Chapter: 10, Displaying the class activation heatmap
    # arr = arr / np.max(arr)

    # Kodelinjen under kommer delvis fra et annet prosjekt gruppen har arbeidet med i faget, DAT255. Referanse: A. Kidess, J. O. Rosberg, og J. Kaste, «Upublisert egen semesteroppgave i faget, DAT255: Explainability methods for image classification», Upublisert egen semesteroppgave i faget, DAT255, Høgskulen på Vestlandet, Haugesund, 2026.
    # Kodelinjen i DAT255-prosjektet kommer i høyeste grad herfra: https://github.com/HVL-ML/DAT255/blob/main/notebooks/03_advanced_image_classification.ipynb.
    arr = keras.ops.expand_dims(arr, 0)

    # Linjen under er delvis basert på: https://www.gradio.app/docs/gradio/checkboxgroup
    return_array = [gr.Image(),gr.Image(),gr.Image(),gr.Image(),gr.Label(),gr.Label(),gr.Image(),gr.Image(),gr.Image(),gr.Image(),gr.Label(),gr.Label(),gr.Image(),gr.Image(),gr.Image(),gr.Image(),gr.Label(),gr.Label(), img, gr.CheckboxGroup()]

    if "Baseline-modell" in checkbox_group:
        model = model1
        # Predict!
        preds = model.predict(arr, verbose=0)
        print(model.predict(arr, verbose=0))
        print(model.predict(arr*255.0, verbose=0))
        print(model.predict(arr/255.0, verbose=0))

        # Linjen under kommer delvis fra et annet prosjekt gruppen har arbeidet med i faget, DAT255. Referanse: A. Kidess, J. O. Rosberg, og J. Kaste, «Upublisert egen semesteroppgave i faget, DAT255: Explainability methods for image classification», Upublisert egen semesteroppgave i faget, DAT255, Høgskulen på Vestlandet, Haugesund, 2026.
        # heatmap = get_heatmap(arr, "lvl1", model)

        # =====
        # Koden under kommer i stor grad fra et annet prosjekt gruppen har arbeidet med i faget, DAT255. Referanse: A. Kidess, J. O. Rosberg, og J. Kaste, «Upublisert egen semesteroppgave i faget, DAT255: Explainability methods for image classification», Upublisert egen semesteroppgave i faget, DAT255, Høgskulen på Vestlandet, Haugesund, 2026.
        # Koden fra DAT255-prosjektet kommer i høyeste grad herfra: SmoothGrad: removing noise by adding noise, 2.2. Smoothing noisy gradients (https://arxiv.org/pdf/1706.03825)
        # og herfra: https://numpy.org/doc/2.1/reference/random/generated/numpy.random.Generator.normal.html#numpy.random.Generator.normal

        rng = np.random.default_rng()
        finished_heatmap_lvl1 = np.zeros(arr[0].shape[:2])
        if (sg_checkbox == True):
            N = 50.0
            for n in range(int(N)):
                noise_map = rng.normal(0, 0.20, arr[0].shape)
                noisy_img = arr * noise_map
                current_heatmap = get_heatmap(noisy_img, "lvl1", model)
                finished_heatmap_lvl1 += current_heatmap
            finished_heatmap_lvl1 /= N
        else:
            finished_heatmap_lvl1 = get_heatmap(arr, "lvl1", model)
        # Kodelinjen under kommer fra et annet prosjekt gruppen har arbeidet med i faget, DAT255. Referanse: A. Kidess, J. O. Rosberg, og J. Kaste, «Upublisert egen semesteroppgave i faget, DAT255: Explainability methods for image classification», Upublisert egen semesteroppgave i faget, DAT255, Høgskulen på Vestlandet, Haugesund, 2026..
        # Idéen til den kodelinjen kommer herfra: SmoothGrad: removing noise by adding noise, Capping outlying values (https://arxiv.org/pdf/1706.03825)
        finished_heatmap_lvl1 = cap_heatmap(finished_heatmap_lvl1)

        # Koden over kommer i stor grad fra et annet prosjekt gruppen har arbeidet med i faget, DAT255. Referanse: A. Kidess, J. O. Rosberg, og J. Kaste, «Upublisert egen semesteroppgave i faget, DAT255: Explainability methods for image classification», Upublisert egen semesteroppgave i faget, DAT255, Høgskulen på Vestlandet, Haugesund, 2026.
        # Koden fra DAT255-prosjektet kommer i høyeste grad herfra: SmoothGrad: removing noise by adding noise, 2.2. Smoothing noisy gradients (https://arxiv.org/pdf/1706.03825)
        # og herfra: https://numpy.org/doc/2.1/reference/random/generated/numpy.random.Generator.normal.html#numpy.random.Generator.normal
        # =====

        # The below line is based on Deep Learning with Python, Third edition (https://deeplearningwithpython.io). Chapter: 10, Visualizing heatmaps of class activation
        superimposed = superimpose(arr[0], finished_heatmap_lvl1)

        # Line below is based on: https://pillow.readthedocs.io/en/stable/reference/Image.html
        superimposed_lvl2 = Image.new(mode="RGB", size=IMG_RES)

        # Line below is based on: https://pillow.readthedocs.io/en/stable/reference/Image.html
        finished_heatmap_lvl2 = Image.new(mode="RGB", size=IMG_RES)
        if preds["lvl1"][0][0] > 0.5:
            # Linjen under kommer delvis fra et annet prosjekt gruppen har arbeidet med i faget, DAT255. Referanse: A. Kidess, J. O. Rosberg, og J. Kaste, «Upublisert egen semesteroppgave i faget, DAT255: Explainability methods for image classification», Upublisert egen semesteroppgave i faget, DAT255, Høgskulen på Vestlandet, Haugesund, 2026.
            # heatmap_lvl2 = get_heatmap(arr, "lvl2", model)

            # =====
            # Koden under kommer i stor grad fra et annet prosjekt gruppen har arbeidet med i faget, DAT255. Referanse: A. Kidess, J. O. Rosberg, og J. Kaste, «Upublisert egen semesteroppgave i faget, DAT255: Explainability methods for image classification», Upublisert egen semesteroppgave i faget, DAT255, Høgskulen på Vestlandet, Haugesund, 2026.
            # Koden fra DAT255-prosjektet kommer i høyeste grad herfra: SmoothGrad: removing noise by adding noise, 2.2. Smoothing noisy gradients (https://arxiv.org/pdf/1706.03825)
            # og herfra: https://numpy.org/doc/2.1/reference/random/generated/numpy.random.Generator.normal.html#numpy.random.Generator.normal

            rng = np.random.default_rng()
            finished_heatmap_lvl2 = np.zeros(arr[0].shape[:2])
            if (sg_checkbox == True):
                N = 50.0
                for n in range(int(N)):
                    noise_map = rng.normal(0, 0.20, arr[0].shape)
                    noisy_img = arr * noise_map
                    # Argmax under er basert på: https://numpy.org/doc/2.2/reference/generated/numpy.argmax.html
                    current_heatmap = get_heatmap(noisy_img, "lvl2", model, np.argmax(preds["lvl2"], 1)[0])
                    finished_heatmap_lvl2 += current_heatmap
                finished_heatmap_lvl2 /= N
            else:
                finished_heatmap_lvl2 = get_heatmap(arr, "lvl1", model)
            # Kodelinjen under kommer fra et annet prosjekt gruppen har arbeidet med i faget, DAT255. Referanse: A. Kidess, J. O. Rosberg, og J. Kaste, «Upublisert egen semesteroppgave i faget, DAT255: Explainability methods for image classification», Upublisert egen semesteroppgave i faget, DAT255, Høgskulen på Vestlandet, Haugesund, 2026..
            # Idéen til den kodelinjen kommer herfra: SmoothGrad: removing noise by adding noise, Capping outlying values (https://arxiv.org/pdf/1706.03825)
            finished_heatmap_lvl2 = cap_heatmap(finished_heatmap_lvl2)

            # Koden over kommer i stor grad fra et annet prosjekt gruppen har arbeidet med i faget, DAT255. Referanse: A. Kidess, J. O. Rosberg, og J. Kaste, «Upublisert egen semesteroppgave i faget, DAT255: Explainability methods for image classification», Upublisert egen semesteroppgave i faget, DAT255, Høgskulen på Vestlandet, Haugesund, 2026.
            # Koden fra DAT255-prosjektet kommer i høyeste grad herfra: SmoothGrad: removing noise by adding noise, 2.2. Smoothing noisy gradients (https://arxiv.org/pdf/1706.03825)
            # og herfra: https://numpy.org/doc/2.1/reference/random/generated/numpy.random.Generator.normal.html#numpy.random.Generator.normal
            # =====

            # The below line is based on Deep Learning with Python, Third edition (https://deeplearningwithpython.io). Chapter: 10, Visualizing heatmaps of class activation
            superimposed_lvl2 = superimpose(arr[0], finished_heatmap_lvl2)

        # Line below is based on https://stackoverflow.com/questions/57253048/scipy-misc-has-no-attribute-imsave and https://www.geeksforgeeks.org/python/getting-started-with-imageio-library-in-python/
        # imageio.imwrite("Neinieg.png", superimposed)

        car_model_preds = preds["lvl2"][0]
        # Convertion to int is from: https://www.w3schools.com/python/numpy/numpy_data_types.asp (Converting Data Type on Existing Arrays)
        return_array[0:6] = [superimposed, np.round(finished_heatmap_lvl1).astype('i'), superimposed_lvl2, np.round(finished_heatmap_lvl2).astype('i'), {"Tesla": preds["lvl1"][0][0], "Ikke-Tesla": 1-preds["lvl1"][0][0]}, {'3 2017–2023': car_model_preds[0], '3 2024–nå': car_model_preds[1], 'S 2012–2015': car_model_preds[2], 'S 2016–nå': car_model_preds[3], 'X': car_model_preds[4], 'Y 2020–2024': car_model_preds[5], 'Y 2025-nå': car_model_preds[6]}]

    if "Egenutviklet modell" in checkbox_group:
        model = model2

        # Predict!
        preds = model.predict(arr, verbose=0)
        print(model.predict(arr, verbose=0))
        print(model.predict(arr*255.0, verbose=0))
        print(model.predict(arr/255.0, verbose=0))

        # Linjen under kommer delvis fra et annet prosjekt gruppen har arbeidet med i faget, DAT255. Referanse: A. Kidess, J. O. Rosberg, og J. Kaste, «Upublisert egen semesteroppgave i faget, DAT255: Explainability methods for image classification», Upublisert egen semesteroppgave i faget, DAT255, Høgskulen på Vestlandet, Haugesund, 2026.
        # heatmap = get_heatmap(arr, "lvl1", model)

        # =====
        # Koden under kommer i stor grad fra et annet prosjekt gruppen har arbeidet med i faget, DAT255. Referanse: A. Kidess, J. O. Rosberg, og J. Kaste, «Upublisert egen semesteroppgave i faget, DAT255: Explainability methods for image classification», Upublisert egen semesteroppgave i faget, DAT255, Høgskulen på Vestlandet, Haugesund, 2026.
        # Koden fra DAT255-prosjektet kommer i høyeste grad herfra: SmoothGrad: removing noise by adding noise, 2.2. Smoothing noisy gradients (https://arxiv.org/pdf/1706.03825)
        # og herfra: https://numpy.org/doc/2.1/reference/random/generated/numpy.random.Generator.normal.html#numpy.random.Generator.normal

        rng = np.random.default_rng()
        finished_heatmap_lvl1 = np.zeros(arr[0].shape[:2])
        if (sg_checkbox == True):
            N = 50.0
            for n in range(int(N)):
                noise_map = rng.normal(0, 0.20, arr[0].shape)
                noisy_img = arr * noise_map
                current_heatmap = get_heatmap(noisy_img, "lvl1", model)
                finished_heatmap_lvl1 += current_heatmap
            finished_heatmap_lvl1 /= N
        else:
            finished_heatmap_lvl1 = get_heatmap(arr, "lvl1", model)
        # Kodelinjen under kommer fra et annet prosjekt gruppen har arbeidet med i faget, DAT255. Referanse: A. Kidess, J. O. Rosberg, og J. Kaste, «Upublisert egen semesteroppgave i faget, DAT255: Explainability methods for image classification», Upublisert egen semesteroppgave i faget, DAT255, Høgskulen på Vestlandet, Haugesund, 2026..
        # Idéen til den kodelinjen kommer herfra: SmoothGrad: removing noise by adding noise, Capping outlying values (https://arxiv.org/pdf/1706.03825)
        finished_heatmap_lvl1 = cap_heatmap(finished_heatmap_lvl1)

        # Koden over kommer i stor grad fra et annet prosjekt gruppen har arbeidet med i faget, DAT255. Referanse: A. Kidess, J. O. Rosberg, og J. Kaste, «Upublisert egen semesteroppgave i faget, DAT255: Explainability methods for image classification», Upublisert egen semesteroppgave i faget, DAT255, Høgskulen på Vestlandet, Haugesund, 2026.
        # Koden fra DAT255-prosjektet kommer i høyeste grad herfra: SmoothGrad: removing noise by adding noise, 2.2. Smoothing noisy gradients (https://arxiv.org/pdf/1706.03825)
        # og herfra: https://numpy.org/doc/2.1/reference/random/generated/numpy.random.Generator.normal.html#numpy.random.Generator.normal
        # =====

        # The below line is based on Deep Learning with Python, Third edition (https://deeplearningwithpython.io). Chapter: 10, Visualizing heatmaps of class activation
        superimposed = superimpose(arr[0], finished_heatmap_lvl1)

        # Line below is based on: https://pillow.readthedocs.io/en/stable/reference/Image.html
        superimposed_lvl2 = Image.new(mode="RGB", size=IMG_RES)

        # Line below is based on: https://pillow.readthedocs.io/en/stable/reference/Image.html
        finished_heatmap_lvl2 = Image.new(mode="RGB", size=IMG_RES)
        if preds["lvl1"][0][0] > 0.5:
            # Linjen under kommer delvis fra et annet prosjekt gruppen har arbeidet med i faget, DAT255. Referanse: A. Kidess, J. O. Rosberg, og J. Kaste, «Upublisert egen semesteroppgave i faget, DAT255: Explainability methods for image classification», Upublisert egen semesteroppgave i faget, DAT255, Høgskulen på Vestlandet, Haugesund, 2026.
            # heatmap_lvl2 = get_heatmap(arr, "lvl2", model)

            # =====
            # Koden under kommer i stor grad fra et annet prosjekt gruppen har arbeidet med i faget, DAT255. Referanse: A. Kidess, J. O. Rosberg, og J. Kaste, «Upublisert egen semesteroppgave i faget, DAT255: Explainability methods for image classification», Upublisert egen semesteroppgave i faget, DAT255, Høgskulen på Vestlandet, Haugesund, 2026.
            # Koden fra DAT255-prosjektet kommer i høyeste grad herfra: SmoothGrad: removing noise by adding noise, 2.2. Smoothing noisy gradients (https://arxiv.org/pdf/1706.03825)
            # og herfra: https://numpy.org/doc/2.1/reference/random/generated/numpy.random.Generator.normal.html#numpy.random.Generator.normal

            rng = np.random.default_rng()
            finished_heatmap_lvl2 = np.zeros(arr[0].shape[:2])
            if (sg_checkbox == True):
                N = 50.0
                for n in range(int(N)):
                    noise_map = rng.normal(0, 0.20, arr[0].shape)
                    noisy_img = arr * noise_map
                    # Argmax under er basert på: https://numpy.org/doc/2.2/reference/generated/numpy.argmax.html
                    current_heatmap = get_heatmap(noisy_img, "lvl2", model, np.argmax(preds["lvl2"], 1)[0])
                    finished_heatmap_lvl2 += current_heatmap
                finished_heatmap_lvl2 /= N
            else:
                finished_heatmap_lvl2 = get_heatmap(arr, "lvl1", model)
            # Kodelinjen under kommer fra et annet prosjekt gruppen har arbeidet med i faget, DAT255. Referanse: A. Kidess, J. O. Rosberg, og J. Kaste, «Upublisert egen semesteroppgave i faget, DAT255: Explainability methods for image classification», Upublisert egen semesteroppgave i faget, DAT255, Høgskulen på Vestlandet, Haugesund, 2026..
            # Idéen til den kodelinjen kommer herfra: SmoothGrad: removing noise by adding noise, Capping outlying values (https://arxiv.org/pdf/1706.03825)
            finished_heatmap_lvl2 = cap_heatmap(finished_heatmap_lvl2)

            # Koden over kommer i stor grad fra et annet prosjekt gruppen har arbeidet med i faget, DAT255. Referanse: A. Kidess, J. O. Rosberg, og J. Kaste, «Upublisert egen semesteroppgave i faget, DAT255: Explainability methods for image classification», Upublisert egen semesteroppgave i faget, DAT255, Høgskulen på Vestlandet, Haugesund, 2026.
            # Koden fra DAT255-prosjektet kommer i høyeste grad herfra: SmoothGrad: removing noise by adding noise, 2.2. Smoothing noisy gradients (https://arxiv.org/pdf/1706.03825)
            # og herfra: https://numpy.org/doc/2.1/reference/random/generated/numpy.random.Generator.normal.html#numpy.random.Generator.normal
            # =====

            # The below line is based on Deep Learning with Python, Third edition (https://deeplearningwithpython.io). Chapter: 10, Visualizing heatmaps of class activation
            superimposed_lvl2 = superimpose(arr[0], finished_heatmap_lvl2)

        # Line below is based on https://stackoverflow.com/questions/57253048/scipy-misc-has-no-attribute-imsave and https://www.geeksforgeeks.org/python/getting-started-with-imageio-library-in-python/
        # imageio.imwrite("Neinieg.png", superimposed)

        car_model_preds = preds["lvl2"][0]
        # Convertion to int is from: https://www.w3schools.com/python/numpy/numpy_data_types.asp (Converting Data Type on Existing Arrays)
        return_array[6:12] = [superimposed, np.round(finished_heatmap_lvl1).astype('i'), superimposed_lvl2, np.round(finished_heatmap_lvl2).astype('i'), {"Tesla": preds["lvl1"][0][0], "Ikke-Tesla": 1-preds["lvl1"][0][0]}, {'3 2017–2023': car_model_preds[0], '3 2024–nå': car_model_preds[1], 'S 2012–2015': car_model_preds[2], 'S 2016–nå': car_model_preds[3], 'X': car_model_preds[4], 'Y 2020–2024': car_model_preds[5], 'Y 2025-nå': car_model_preds[6]}]

    if "Fine-tuned pretrent modell" in checkbox_group:
        model = model3
        # Predict!
        preds = model.predict(arr, verbose=0)
        print(model.predict(arr, verbose=0))
        print(model.predict(arr*255.0, verbose=0))
        print(model.predict(arr/255.0, verbose=0))

        # Linjen under kommer delvis fra et annet prosjekt gruppen har arbeidet med i faget, DAT255. Referanse: A. Kidess, J. O. Rosberg, og J. Kaste, «Upublisert egen semesteroppgave i faget, DAT255: Explainability methods for image classification», Upublisert egen semesteroppgave i faget, DAT255, Høgskulen på Vestlandet, Haugesund, 2026.
        # heatmap = get_heatmap(arr, "lvl1", model)

        # =====
        # Koden under kommer i stor grad fra et annet prosjekt gruppen har arbeidet med i faget, DAT255. Referanse: A. Kidess, J. O. Rosberg, og J. Kaste, «Upublisert egen semesteroppgave i faget, DAT255: Explainability methods for image classification», Upublisert egen semesteroppgave i faget, DAT255, Høgskulen på Vestlandet, Haugesund, 2026.
        # Koden fra DAT255-prosjektet kommer i høyeste grad herfra: SmoothGrad: removing noise by adding noise, 2.2. Smoothing noisy gradients (https://arxiv.org/pdf/1706.03825)
        # og herfra: https://numpy.org/doc/2.1/reference/random/generated/numpy.random.Generator.normal.html#numpy.random.Generator.normal

        rng = np.random.default_rng()
        finished_heatmap_lvl1 = np.zeros(arr[0].shape[:2])
        if (sg_checkbox == True):
            N = 50.0
            for n in range(int(N)):
                noise_map = rng.normal(0, 0.20, arr[0].shape)
                noisy_img = arr * noise_map
                current_heatmap = get_heatmap(noisy_img, "lvl1", model)
                finished_heatmap_lvl1 += current_heatmap
            finished_heatmap_lvl1 /= N
        else:
            finished_heatmap_lvl1 = get_heatmap(arr, "lvl1", model)
        # Kodelinjen under kommer fra et annet prosjekt gruppen har arbeidet med i faget, DAT255. Referanse: A. Kidess, J. O. Rosberg, og J. Kaste, «Upublisert egen semesteroppgave i faget, DAT255: Explainability methods for image classification», Upublisert egen semesteroppgave i faget, DAT255, Høgskulen på Vestlandet, Haugesund, 2026..
        # Idéen til den kodelinjen kommer herfra: SmoothGrad: removing noise by adding noise, Capping outlying values (https://arxiv.org/pdf/1706.03825)
        finished_heatmap_lvl1 = cap_heatmap(finished_heatmap_lvl1)

        # Koden over kommer i stor grad fra et annet prosjekt gruppen har arbeidet med i faget, DAT255. Referanse: A. Kidess, J. O. Rosberg, og J. Kaste, «Upublisert egen semesteroppgave i faget, DAT255: Explainability methods for image classification», Upublisert egen semesteroppgave i faget, DAT255, Høgskulen på Vestlandet, Haugesund, 2026.
        # Koden fra DAT255-prosjektet kommer i høyeste grad herfra: SmoothGrad: removing noise by adding noise, 2.2. Smoothing noisy gradients (https://arxiv.org/pdf/1706.03825)
        # og herfra: https://numpy.org/doc/2.1/reference/random/generated/numpy.random.Generator.normal.html#numpy.random.Generator.normal
        # =====

        # The below line is based on Deep Learning with Python, Third edition (https://deeplearningwithpython.io). Chapter: 10, Visualizing heatmaps of class activation
        superimposed = superimpose(arr[0], finished_heatmap_lvl1)

        # Line below is based on: https://pillow.readthedocs.io/en/stable/reference/Image.html
        superimposed_lvl2 = Image.new(mode="RGB", size=IMG_RES)

        # Line below is based on: https://pillow.readthedocs.io/en/stable/reference/Image.html
        finished_heatmap_lvl2 = Image.new(mode="RGB", size=IMG_RES)
        if preds["lvl1"][0][0] > 0.5:
            # Linjen under kommer delvis fra et annet prosjekt gruppen har arbeidet med i faget, DAT255. Referanse: A. Kidess, J. O. Rosberg, og J. Kaste, «Upublisert egen semesteroppgave i faget, DAT255: Explainability methods for image classification», Upublisert egen semesteroppgave i faget, DAT255, Høgskulen på Vestlandet, Haugesund, 2026.
            # heatmap_lvl2 = get_heatmap(arr, "lvl2", model)

            # =====
            # Koden under kommer i stor grad fra et annet prosjekt gruppen har arbeidet med i faget, DAT255. Referanse: A. Kidess, J. O. Rosberg, og J. Kaste, «Upublisert egen semesteroppgave i faget, DAT255: Explainability methods for image classification», Upublisert egen semesteroppgave i faget, DAT255, Høgskulen på Vestlandet, Haugesund, 2026.
            # Koden fra DAT255-prosjektet kommer i høyeste grad herfra: SmoothGrad: removing noise by adding noise, 2.2. Smoothing noisy gradients (https://arxiv.org/pdf/1706.03825)
            # og herfra: https://numpy.org/doc/2.1/reference/random/generated/numpy.random.Generator.normal.html#numpy.random.Generator.normal

            rng = np.random.default_rng()
            finished_heatmap_lvl2 = np.zeros(arr[0].shape[:2])
            if (sg_checkbox == True):
                N = 50.0
                for n in range(int(N)):
                    noise_map = rng.normal(0, 0.20, arr[0].shape)
                    noisy_img = arr * noise_map
                    # Argmax under er basert på: https://numpy.org/doc/2.2/reference/generated/numpy.argmax.html
                    current_heatmap = get_heatmap(noisy_img, "lvl2", model, np.argmax(preds["lvl2"], 1)[0])
                    finished_heatmap_lvl2 += current_heatmap
                finished_heatmap_lvl2 /= N
            else:
                finished_heatmap_lvl2 = get_heatmap(arr, "lvl1", model)
            # Kodelinjen under kommer fra et annet prosjekt gruppen har arbeidet med i faget, DAT255. Referanse: A. Kidess, J. O. Rosberg, og J. Kaste, «Upublisert egen semesteroppgave i faget, DAT255: Explainability methods for image classification», Upublisert egen semesteroppgave i faget, DAT255, Høgskulen på Vestlandet, Haugesund, 2026..
            # Idéen til den kodelinjen kommer herfra: SmoothGrad: removing noise by adding noise, Capping outlying values (https://arxiv.org/pdf/1706.03825)
            finished_heatmap_lvl2 = cap_heatmap(finished_heatmap_lvl2)

            # Koden over kommer i stor grad fra et annet prosjekt gruppen har arbeidet med i faget, DAT255. Referanse: A. Kidess, J. O. Rosberg, og J. Kaste, «Upublisert egen semesteroppgave i faget, DAT255: Explainability methods for image classification», Upublisert egen semesteroppgave i faget, DAT255, Høgskulen på Vestlandet, Haugesund, 2026.
            # Koden fra DAT255-prosjektet kommer i høyeste grad herfra: SmoothGrad: removing noise by adding noise, 2.2. Smoothing noisy gradients (https://arxiv.org/pdf/1706.03825)
            # og herfra: https://numpy.org/doc/2.1/reference/random/generated/numpy.random.Generator.normal.html#numpy.random.Generator.normal
            # =====

            # The below line is based on Deep Learning with Python, Third edition (https://deeplearningwithpython.io). Chapter: 10, Visualizing heatmaps of class activation
            superimposed_lvl2 = superimpose(arr[0], finished_heatmap_lvl2)
        # Line below is based on https://stackoverflow.com/questions/57253048/scipy-misc-has-no-attribute-imsave and https://www.geeksforgeeks.org/python/getting-started-with-imageio-library-in-python/
        # imageio.imwrite("Neinieg.png", superimposed)

        car_model_preds = preds["lvl2"][0]
        # Convertion to int is from: https://www.w3schools.com/python/numpy/numpy_data_types.asp (Converting Data Type on Existing Arrays)
        return_array[12:18] = [superimposed, np.round(finished_heatmap_lvl1).astype('i'), superimposed_lvl2, np.round(finished_heatmap_lvl2).astype('i'), {"Tesla": preds["lvl1"][0][0], "Ikke-Tesla": 1-preds["lvl1"][0][0]}, {'3 2017–2023': car_model_preds[0], '3 2024–nå': car_model_preds[1], 'S 2012–2015': car_model_preds[2], 'S 2016–nå': car_model_preds[3], 'X': car_model_preds[4], 'Y 2020–2024': car_model_preds[5], 'Y 2025-nå': car_model_preds[6]}]
    return return_array



# #### Cellen under brukte elementer herfra: https://github.com/gradio-app/gradio/issues/2066 og https://discuss.huggingface.co/t/how-to-programmatically-enable-or-disable-components/52350

# In[9]:


def change_visibility(checkbox_group, model_1_label1, model_1_label2, model_2_label1, model_2_label2, model_3_label1, model_3_label2):
    return_array = [gr.Image(),gr.Image(),gr.Image(),gr.Image(),gr.Label(),gr.Label(),gr.Markdown(),gr.Image(),gr.Image(),gr.Image(),gr.Image(),gr.Label(),gr.Label(),gr.Markdown(),gr.Image(),gr.Image(),gr.Image(),gr.Image(),gr.Label(),gr.Label(),gr.Markdown()]
    if "Baseline-modell" in checkbox_group:
        try:
            model_1_label1["Tesla"]
            # The line below is based on: https://www.geeksforgeeks.org/python/python-get-key-with-maximum-value-in-dictionary/
            predicted_model = max(model_1_label2, key=model_1_label2.get)
            return_array[0:7] = [gr.Image(label="Heatmap som viser hvorfor Tesla", visible=True), gr.Image(label="Gråskala-heatmap som viser hvorfor Tesla", visible=True), gr.Image(visible=True, label="Heatmap som viser hvorfor " + predicted_model), gr.Image(visible=True, label="Gråskala-heatmap som viser hvorfor " + predicted_model), gr.Label(visible=True), gr.Label(visible=True), gr.Markdown(visible=True)]
        except:
            return_array[0:7] = [gr.Image(label="Heatmap som viser hvorfor ikke-Tesla", visible=True), gr.Image(label="Gråskala-heatmap som viser hvorfor ikke-Tesla", visible=True), gr.Image(visible=False), gr.Image(visible=False), gr.Label(visible=True), gr.Label(visible=False), gr.Markdown(visible=True)]
    else: 
        return_array[0:7] = [gr.Image(visible=False), gr.Image(visible=False), gr.Image(visible=False), gr.Image(visible=False), gr.Label(visible=False), gr.Label(visible=False), gr.Markdown(visible=False)]
    if "Egenutviklet modell" in checkbox_group:
        try:
            model_2_label1["Tesla"]
            # The line below is based on: https://www.geeksforgeeks.org/python/python-get-key-with-maximum-value-in-dictionary/
            predicted_model = max(model_2_label2, key=model_2_label2.get)
            return_array[7:14] = [gr.Image(label="Heatmap som viser hvorfor Tesla", visible=True), gr.Image(label="Gråskala-heatmap som viser hvorfor Tesla", visible=True), gr.Image(visible=True, label="Heatmap som viser hvorfor " + predicted_model), gr.Image(visible=True, label="Gråskala-heatmap som viser hvorfor " + predicted_model), gr.Label(visible=True), gr.Label(visible=True), gr.Markdown(visible=True)]
        except:
            return_array[7:14] = [gr.Image(label="Heatmap som viser hvorfor ikke-Tesla", visible=True), gr.Image(label="Gråskala-heatmap som viser hvorfor ikke-Tesla", visible=True), gr.Image(visible=False), gr.Image(visible=False), gr.Label(visible=True), gr.Label(visible=False), gr.Markdown(visible=True)]
    else: 
        return_array[7:14] = [gr.Image(visible=False), gr.Image(visible=False), gr.Image(visible=False), gr.Image(visible=False), gr.Label(visible=False), gr.Label(visible=False), gr.Markdown(visible=False)]
    if "Fine-tuned pretrent modell" in checkbox_group:
        try:
            model_3_label1["Tesla"]
            # The line below is based on: https://www.geeksforgeeks.org/python/python-get-key-with-maximum-value-in-dictionary/
            predicted_model = max(model_3_label2, key=model_3_label2.get)
            return_array[14:21] = [gr.Image(label="Heatmap som viser hvorfor Tesla", visible=True), gr.Image(label="Gråskala-heatmap som viser hvorfor Tesla", visible=True), gr.Image(visible=True, label="Heatmap som viser hvorfor " + predicted_model), gr.Image(visible=True, label="Gråskala-heatmap som viser hvorfor " + predicted_model), gr.Label(visible=True), gr.Label(visible=True), gr.Markdown(visible=True)]
        except:
            return_array[14:21] = [gr.Image(label="Heatmap som viser hvorfor ikke-Tesla", visible=True), gr.Image(label="Gråskala-heatmap som viser hvorfor ikke-Tesla", visible=True), gr.Image(visible=False), gr.Image(visible=False), gr.Label(visible=True), gr.Label(visible=False), gr.Markdown(visible=True)]
    else: 
        return_array[14:21] = [gr.Image(visible=False), gr.Image(visible=False), gr.Image(visible=False), gr.Image(visible=False), gr.Label(visible=False), gr.Label(visible=False), gr.Markdown(visible=False)]
    return return_array


# ## Set up the Gradio interface
# 
# Check the [documentation](https://www.gradio.app/docs) for the various things we can add here.

# In[10]:


# Example images
examples = [
    "https://cdn.britannica.com/79/232779-050-6B0411D7/German-Shepherd-dog-Alsatian.jpg",
    "https://cdn.britannica.com/41/126641-050-E1CA0E61/cat-suns-hill-Parthenon-Athens-Greece-Acropolis.jpg",
]

with gr.Blocks(title="Multiattributt visuell kjøretøygjenkjenning under varierende lysforhold") as demo:
    gr.Markdown("## Multiattributt visuell kjøretøygjenkjenning under varierende lysforhold")
    gr.Markdown(
        "Last opp et bilde av et kjøretøy."
    )
    with gr.Row():
        components = [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],0,0,0]
        components[3] = gr.Image(type="pil", label="Opplastet bilde", height=400)
        # Kodelinjen under er basert på: https://www.gradio.app/docs/gradio/checkboxgroup
        components[4] = gr.CheckboxGroup(choices=["Egenutviklet modell", "Fine-tuned pretrent modell"])
        # components[4] = gr.CheckboxGroup(choices=["Baseline-modell", "Egenutviklet modell", "Fine-tuned pretrent modell"])

    components[5] = gr.Checkbox("SmoothGrad", label="Utjevnede heatmaps (SmoothGrad)")

    # Linjen under er basert på: https://www.gradio.app/docs/gradio/blocks
    components[0][6] = gr.Markdown("Prediksjonene til baseline-modell:", visible=False)
    with gr.Row():
        # Linjen under er basert på: https://www.gradio.app/4.44.1/guides/controlling-layout
        with gr.Column():
            # De to linjene under er delvis basert på: https://github.com/gradio-app/gradio/issues/10394
            components[0][0] = gr.Image(type="pil", label="Heatmap", format="png", visible=False)
            components[0][1] = gr.Image(type="pil", label="Heatmap", format="png", visible=False)
        # Linjen under er basert på: https://www.gradio.app/4.44.1/guides/controlling-layout
        with gr.Column():
            # De to linjene under er delvis basert på: https://github.com/gradio-app/gradio/issues/10394
            components[0][2] = gr.Image(type="pil", label="Heatmap", format="png", visible=False)
            components[0][3] = gr.Image(type="pil", label="Heatmap", format="png", visible=False)
        # Linjen under er basert på: https://www.gradio.app/4.44.1/guides/controlling-layout
        with gr.Column():
            components[0][4] = gr.Label(num_top_classes=1, label="Bilmerke", visible=False)
            components[0][5] = gr.Label(num_top_classes=3, label="Bilmodell", visible=False)

    # Linjen under er basert på: https://www.gradio.app/docs/gradio/blocks
    components[1][6] = gr.Markdown("Prediksjonene til egenutviklet modell:", visible=False)
    with gr.Row():
        # Linjen under er basert på: https://www.gradio.app/4.44.1/guides/controlling-layout
        with gr.Column():
            # De to linjene under er delvis basert på: https://github.com/gradio-app/gradio/issues/10394
            components[1][0] = gr.Image(type="pil", label="Heatmap", format="png", visible=False)
            components[1][1] = gr.Image(type="pil", label="Heatmap", format="png", visible=False)
        # Linjen under er basert på: https://www.gradio.app/4.44.1/guides/controlling-layout
        with gr.Column():
            # De to linjene under er delvis basert på: https://github.com/gradio-app/gradio/issues/10394
            components[1][2] = gr.Image(type="pil", label="Heatmap", format="png", visible=False)
            components[1][3] = gr.Image(type="pil", label="Heatmap", format="png", visible=False)
        # Linjen under er basert på: https://www.gradio.app/4.44.1/guides/controlling-layout
        with gr.Column():
            components[1][4] = gr.Label(num_top_classes=1, label="Bilmerke", visible=False)
            components[1][5] = gr.Label(num_top_classes=3, label="Bilmodell", visible=False)

    # Linjen under er basert på: https://www.gradio.app/docs/gradio/blocks
    components[2][6] = gr.Markdown("Prediksjonene til fine-tuned pretrent modell:", visible=False)
    with gr.Row():
        # Linjen under er basert på: https://www.gradio.app/4.44.1/guides/controlling-layout
        with gr.Column():
            # De to linjene under er delvis basert på: https://github.com/gradio-app/gradio/issues/10394
            components[2][0] = gr.Image(type="pil", label="Heatmap", format="png", visible=False)
            components[2][1] = gr.Image(type="pil", label="Heatmap", format="png", visible=False)
        # Linjen under er basert på: https://www.gradio.app/4.44.1/guides/controlling-layout
        with gr.Column():
            # De to linjene under er delvis basert på: https://github.com/gradio-app/gradio/issues/10394
            components[2][2] = gr.Image(type="pil", label="Heatmap", format="png", visible=False)
            components[2][3] = gr.Image(type="pil", label="Heatmap", format="png", visible=False)
        # Linjen under er basert på: https://www.gradio.app/4.44.1/guides/controlling-layout
        with gr.Column():
            components[2][4] = gr.Label(num_top_classes=1, label="Bilmerke", visible=False)
            components[2][5] = gr.Label(num_top_classes=3, label="Bilmodell", visible=False)

    classify_btn = gr.Button("Classify", variant="primary")
    classify_btn.click(fn=classify_image, inputs=[components[3],components[4], components[5]], outputs=[components[0][0],components[0][1],components[0][2],components[0][3],components[0][4],components[0][5],components[1][0],components[1][1],components[1][2],components[1][3],components[1][4],components[1][5],components[2][0],components[2][1],components[2][2],components[2][3],components[2][4],components[2][5], components[3], components[4]])
    # Linjen under er basert på: https://github.com/gradio-app/gradio/issues/2066 og https://discuss.huggingface.co/t/how-to-programmatically-enable-or-disable-components/52350
    classify_btn.click(fn=change_visibility, inputs=[components[4], components[0][4], components[0][5], components[1][4], components[1][5], components[2][4], components[2][5]], outputs=[components[0][0], components[0][1], components[0][2], components[0][3], components[0][4], components[0][5], components[0][6], components[1][0], components[1][1], components[1][2], components[1][3], components[1][4], components[1][5], components[1][6], components[2][0], components[2][1], components[2][2], components[2][3], components[2][4], components[2][5], components[2][6]])

    # examples = gr.Examples(
    #     examples=examples,
    #     inputs=components[3],
    #     # outputs=output,
    #     # fn=classify_image,
    #     # cache_examples=True
    # )


# Now we can run it:

# In[11]:


demo.launch(share=False)

