import pandas as pd
from typing import Tuple
import os
from keras.utils import load_img, img_to_array
import sys
import re
import matplotlib.pyplot as plt


internal = pd.read_csv('annotations/internal.csv')
external = pd.read_csv('annotations/external.csv')


def image_label_to_array(image_label: str, target_size: Tuple[int, int]):
    image_name = image_label.split("/")[-1].split("5C")[-1]
    image_name = re.sub(r"^.{8}-", "", image_name)
    path = os.path.join("datasett", image_name)
    img = load_img(path, target_size=target_size,
                   interpolation="bicubic", keep_aspect_ratio=True)
    img = img_to_array(img)/255
    return img


def read_internal_data(target_size: Tuple[int, int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    internal_labels = internal.drop("image", axis=1)
    internal_images = pd.DataFrame(
        [image_label_to_array(i, target_size=target_size)
         for i in internal["image"]]
    )
    return (internal_labels, internal_images)


def main():
    import time
    start_time = time.perf_counter()

    imgs = []
    for img in internal["image"]:
        imgs.append(image_label_to_array(img, (300, 300)))
    print(len(imgs))
    print(sys.getsizeof(imgs))
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.4f} seconds")


if __name__ == '__main__':
    main()
