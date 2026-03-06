from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from typing import Tuple
import os
from keras.utils import load_img, img_to_array
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
IMG_ROOT = BASE_DIR / 'datasett'
IMG_ROOT_ANDRE = BASE_DIR / 'datasett_src'


internal = pd.read_csv(BASE_DIR / 'annotations/internal_new.csv')
external = pd.read_csv(BASE_DIR / 'annotations/external_new.csv')


def image_label_to_array(image_label: str, target_size: Tuple[int, int], img_root=IMG_ROOT):
    image_name = image_label
    path = os.path.join(BASE_DIR, img_root, image_name)
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


def load_images_and_labels(target_size, data=internal, img_root=IMG_ROOT):
    images_list = []
    valid_indices = []

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda img_label: image_label_to_array(
            img_label, target_size, img_root=img_root), data['image']))

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


def read_andre_data(target_size=(300, 300)):
    train_df = pd.read_csv(BASE_DIR / 'datasplitt/train.csv')
    val_df = pd.read_csv(BASE_DIR / 'datasplitt/val.csv')
    test_df = pd.read_csv(BASE_DIR / 'datasplitt/test.csv')
    le_gate1 = LabelEncoder()
    le_gate1.fit(train_df['lvl1'])
    le_gate2 = LabelEncoder()
    le_gate2.fit(train_df.iloc[train_df['lvl1'] == 'Tesla', 'lvl2'])
    xs = []
    for df in [train_df, val_df, test_df]:
        df['gate1'] = le_gate1.transform(df['lvl1'])
        df['gate2'] = le_gate2.transform(df['lvl2'])
        df = fix_image_paths(df, IMG_ROOT_ANDRE)
        xs.append(load_images_and_labels(
            data=df, target_size=target_size, img_root=IMG_ROOT_ANDRE))

    return xs[0], xs[1], xs[2]


def read_stratified_data(
    val_size: float = 0.15,
    test_size: float = 0.15,
    target_size: Tuple[int, int] = (300, 300),
    columns=('color', 'lighting', 'model', 'year'),
    strata_threshold=10


) -> Tuple:
    combined = pd.concat([internal, external]).reset_index(drop=True)

    combined['tmp_strata'] = combined[list(columns)].fillna(
        '').astype(str).agg('-'.join, axis=1)

    strata_counts = combined['tmp_strata'].value_counts()
    rare_strata_labels = strata_counts[strata_counts < strata_threshold].index

    rare_mask = combined['tmp_strata'].isin(rare_strata_labels)
    under_represented_rows = combined[rare_mask].copy()
    safe_combined = combined[~rare_mask].copy()

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

    train_df = pd.concat([train_df, under_represented_rows], ignore_index=True)

    for df in [train_df, val_df, test_df]:
        df.drop(columns=['tmp_strata'], inplace=True)

    train_x, train_y = load_images_and_labels(
        target_size=target_size, data=train_df)
    val_x, val_y = load_images_and_labels(target_size=target_size, data=val_df)
    test_x, test_y = load_images_and_labels(
        target_size=target_size, data=test_df)

    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


def fix_image_paths(df, root_dir):
    root = Path(root_dir)
    fixed_count = 0
    missing_count = 0

    def find_correct_extension(row_path):
        nonlocal fixed_count, missing_count
        p = root / row_path

        # 1. If it already exists, we're good
        if p.exists():
            return row_path

        # 2. Try common alternative extensions
        # (e.g., if CSV says .jpg, try .jpeg, .JPG, .png, etc.)
        alternatives = ['.jpeg', '.jpg', '.JPG', '.JPEG', '.png']
        for ext in alternatives:
            alt_p = p.with_suffix(ext)
            if alt_p.exists():
                fixed_count += 1
                # Return the path relative to IMG_ROOT
                return str(alt_p.relative_to(root))

        # 3. If still not found, log it
        missing_count += 1
        return row_path

    # Apply the fix to the "image" column
    df['image'] = df['image'].apply(find_correct_extension)

    print(f"Correction complete!")
    print(f" - Files found as-is: {len(df) - fixed_count - missing_count}")
    print(f" - Extensions corrected: {fixed_count}")
    print(f" - Still missing (not found): {missing_count}")

    return df


def main():
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = read_andre_data()
    for pair in [(train_x, train_y), (val_x, val_y), (test_x, test_y)]:
        print(pair[0].shape)
        print(pair[1].shape)


if __name__ == '__main__':
    main()
