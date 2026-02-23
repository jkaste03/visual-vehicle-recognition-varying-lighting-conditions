from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from typing import Tuple
import os
from keras.utils import load_img, img_to_array
import re
from sklearn.model_selection import train_test_split
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
labels = ['color', 'image', 'lighting',
          'model', 'year_3', 'year_s', 'year_x', 'year_y']
internal = pd.read_csv(base_dir / 'annotations/internal.csv')
internal = internal[labels]

external = pd.read_csv(base_dir / 'annotations/external.csv')
external = external[labels]

# Linjene under er basert på: https://stackoverflow.com/questions/71523769/merge-two-columns-into-one-with-nan-values
internal['year'] = internal[['year_3', 'year_s', 'year_x', 'year_y']].bfill(
    axis='columns').iloc[:, 0]
external['year'] = external[['year_3', 'year_s', 'year_x', 'year_y']].bfill(
    axis='columns').iloc[:, 0]

# De to linjene under er basert på: https://stackoverflow.com/questions/40389018/dropping-multiple-columns-from-a-dataframe
internal.drop(['year_3', 'year_s', 'year_x', 'year_y'], axis=1, inplace=True)
external.drop(['year_3', 'year_s', 'year_x', 'year_y'], axis=1, inplace=True)


filter = (internal['model'] == 'Y') | (internal['model'] == '3')
internal.loc[filter, 'model'] = internal.loc[filter,
                                             'model'] + ' ' + internal.loc[filter, 'year']
filter = (external['model'] == 'Y') | (external['model'] == '3')
external.loc[filter, 'model'] = external.loc[filter,
                                             'model'] + ' ' + external.loc[filter, 'year']

filter = (internal['model'] == 'S') & (internal['year'] == '2012–2015')
internal.loc[filter, 'model'] = 'S 2012–2015'
filter = (external['model'] == 'S') & (external['year'] == '2012–2015')
external.loc[filter, 'model'] = 'S 2012–2015'

filter = internal['model'] == 'S'
internal.loc[filter, 'model'] = 'S 2016–nå'
filter = external['model'] == 'S'
external.loc[filter, 'model'] = 'S 2016–nå'

internal['source'] = 'internal'
external['source'] = 'external'


def image_label_to_array(image_label: str, target_size: Tuple[int, int]):
    image_name = image_label.split('/')[-1].split('5C')[-1]
    image_name = re.sub(r'^.{8}-', '', image_name)
    image_name = re.sub(r'.webp', '.jpg', image_name)
    if ('%25~' in image_name):
        image_name = re.sub(r'25', '', image_name, count=1)

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


def read_data() -> Tuple[Tuple[np.ndarray, pd.DataFrame], Tuple[np.ndarray, pd.DataFrame]]:
    combined = pd.concat([internal, external])
    train_set, test_set = train_test_split(
        combined, test_size=0.15, stratify=combined['model'])
    train_x, train_y = load_images_and_labels(
        target_size=(300, 300), data=train_set)
    test_x, test_y = load_images_and_labels(
        target_size=(300, 300), data=test_set)
    return (train_x, train_y), (test_x, test_y)


def read_stratified_data(test_size: float = 0.15,
                         target_size: Tuple[int, int] = (300, 300),
                         columns=('color', 'lighting', 'model', 'year'),
                         strata_threshold=38
                         ) -> Tuple[Tuple[np.ndarray, pd.DataFrame], Tuple[np.ndarray, pd.DataFrame]]:

    combined = pd.concat([internal, external])

    strata = combined[list(columns)]\
        .fillna('')\
        .astype(str)\
        .agg('-'.join, axis=1)

    strata_count = strata.value_counts()
    under_represented_labels = [
        label for label, count in strata_count.items() if count < strata_threshold
    ]
    under_represented_rows = combined[strata.isin(under_represented_labels)]
    combined = combined[~(strata.isin(under_represented_labels))]
    strata = strata[~(strata.isin(under_represented_labels))]

    train_set, test_set = train_test_split(
        combined, test_size=test_size, stratify=strata)

    train_set = pd.concat([train_set, under_represented_rows])
    train_x, train_y = load_images_and_labels(
        target_size=target_size, data=train_set)
    test_x, test_y = load_images_and_labels(
        target_size=target_size, data=test_set)
    return (train_x, train_y), (test_x, test_y)


def read_gate_one_data() -> Tuple[Tuple[np.ndarray, pd.DataFrame], Tuple[np.ndarray, pd.DataFrame]]:
    (train_x, train_y), (test_x, test_y) = read_data()

    def t(x):
        if x == 'Other car':
            return 1
        return 0

    vt = np.vectorize(t)

    train_y_encoded = vt(train_y['model'])
    test_y_encoded = vt(test_y['model'])
    return (train_x, train_y_encoded), (test_x, test_y_encoded)


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


def main():
    columns = ("model", "lighting")
    read_stratified_data(columns=columns)


if __name__ == '__main__':
    main()
