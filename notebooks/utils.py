from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from typing import Tuple
import os
from keras.utils import load_img, img_to_array
from sklearn.model_selection import train_test_split
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent

internal = pd.read_csv(base_dir / 'annotations/internal_new.csv')
external = pd.read_csv(base_dir / 'annotations/external_new.csv')


def image_label_to_array(image_label: str, target_size: Tuple[int, int]):
    image_name = image_label
    path = os.path.join(base_dir, 'datasett', image_name)
    if (os.path.exists(path)):
        img = load_img(
            path,
            target_size=target_size,
            interpolation='bicubic',
            color_mode='rgb',
            keep_aspect_ratio=True
        )

        img = img_to_array(img) / 255.0
        return img
    print(path)
    return None


def load_images_and_labels(target_size, data=internal):
    images_list = []
    valid_indices = []

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda img_label: image_label_to_array(
            img_label, target_size), data['image']))

    for idx, img in enumerate(results):
        if img is not None:
            images_list.append(img)
            valid_indices.append(idx)

    return np.array(images_list), data.iloc[valid_indices].drop('image', axis=1)


def model_str_to_int(model_str):
    dict = {
        'S 2012–2015': 0, 'S 2016–nå': 1,
        '3 2017–2023': 2, '3 2024–nå': 3,
        'X': 4,
        'Y 2020–2024': 5, 'Y 2025-nå': 6
    }
    return dict[model_str]


def int_to_model_str(x):
    dict = {
        0: 'S 2012–2015',  1: 'S 2016–nå',
        2: '3 2017–2023', 3: '3 2024–nå',
        4: 'X',
        5: 'Y 2020–2024', 6: 'Y 2025-nå'
    }
    return dict[x]


def read_stratified_data(
    val_size: float = 0.15,
    test_size: float = 0.15,
    target_size: Tuple[int, int] = (300, 300),
    columns=('color', 'lighting', 'model', 'year'),
    strata_threshold=10
) -> Tuple:

    combined = pd.concat([internal, external])
    strata = combined[list(columns)].fillna(
        '').astype(str).agg('-'.join, axis=1)

    strata_count = strata.value_counts()
    under_represented_labels = [
        label for label, count in strata_count.items() if count < strata_threshold]

    under_represented_rows = combined[strata.isin(under_represented_labels)]
    safe_combined = combined[~(strata.isin(under_represented_labels))]
    safe_strata = strata[~(strata.isin(under_represented_labels))]

    combined['tmp_strata'] = strata

    strata_count = combined['tmp_strata'].value_counts()
    under_represented_mask = combined['tmp_strata'].isin(
        [label for label, count in strata_count.items() if count < strata_threshold]
    )

    under_represented_rows = combined[under_represented_mask]
    safe_combined = combined[~under_represented_mask]

    train_val_df, test_df = train_test_split(
        safe_combined,
        test_size=test_size,
        random_state=42,
        stratify=safe_combined['tmp_strata']
    )

    relative_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=42,
        stratify=train_val_df['tmp_strata']
    )

    train_df = pd.concat([train_df, under_represented_rows])

    for d in [train_df, val_df, test_df]:
        d.drop(columns=['tmp_strata'], inplace=True)

    train_x, train_y = load_images_and_labels(
        target_size=target_size, data=train_df
    )
    val_x, val_y = load_images_and_labels(target_size=target_size, data=val_df)
    test_x, test_y = load_images_and_labels(
        target_size=target_size,
        data=test_df
    )

    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


def main():
    print(internal)


if __name__ == '__main__':
    main()
