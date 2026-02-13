import numpy as np
import pandas as pd
from typing import Tuple
import os
from keras.utils import load_img, img_to_array
import re
from sklearn.model_selection import train_test_split

internal = pd.read_csv('annotations/internal.csv')
external = pd.read_csv('annotations/external.csv')
default = internal["image"][0]


def image_label_to_array(image_label: str, target_size: Tuple[int, int]):
    image_name = image_label.split("/")[-1].split("5C")[-1]
    image_name = re.sub(r"^.{8}-", "", image_name)
    path = os.path.join("datasett", image_name)
    if (os.path.exists(path)):
        img = load_img(
            path,
            target_size=target_size,
            interpolation="bicubic",
            color_mode="rgb",
            keep_aspect_ratio=True
        )

        img = img_to_array(img) / 255.0

        return img
    return None


# def read_internal_data(target_size: Tuple[int, int], data=internal) -> Tuple[np.ndarray, pd.DataFrame]:
#     internal_labels = data.drop("image", axis=1)

#     internal_images = np.array([
#         image_label_to_array(i, target_size=target_size)
#         for i in internal["image"]
#     ])

#     return (internal_images, internal_labels)


def read_and_split(target_size: Tuple[int, int], data=internal) -> Tuple[np.ndarray, pd.DataFrame]:
    images_list = []
    valid_indices = []

    for idx, img_label in enumerate(data["image"]):
        img = image_label_to_array(img_label, target_size=target_size)

        if img is not None:
            images_list.append(img)
            valid_indices.append(idx)

    external_images = np.array(images_list)
    external_labels = data.iloc[valid_indices].drop("image", axis=1)
    return (external_images, external_labels)


def read_data():
    combined = pd.concat([internal, external])
    train_set, test_set = train_test_split(
        combined, test_size=0.15, stratify=combined['model'])
    train_x, train_y = read_and_split(
        target_size=(300, 300), data=train_set)
    test_x, test_y = read_and_split(target_size=(300, 300), data=test_set)
    return (train_x, train_y), (test_x, test_y)


def main():
    read_data()


if __name__ == '__main__':
    main()
