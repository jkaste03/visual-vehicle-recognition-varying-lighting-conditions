from pathlib import Path

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder

base_dir = Path(__file__).resolve().parent.parent
labels = ['color', 'image', 'lighting',
          'model', 'year_3', 'year_s', 'year_x', 'year_y']
internal = pd.read_csv(base_dir / 'annotations/internal.csv')
internal = internal[labels]

external = pd.read_csv(base_dir / 'annotations/external.csv')
external = external[labels]

internal['year'] = internal[['year_3', 'year_s', 'year_x', 'year_y']].bfill(
    axis='columns').iloc[:, 0]
external['year'] = external[['year_3', 'year_s', 'year_x', 'year_y']].bfill(
    axis='columns').iloc[:, 0]
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


def clean_image_label(image_label):
    image_name = image_label.split('/')[-1].split('5C')[-1]
    image_name = re.sub(r'^.{8}-', '', image_name)
    image_name = re.sub(r'.webp', '.jpg', image_name)
    if ('%25~' in image_name):
        image_name = re.sub(r'25', '', image_name, count=1)
    return image_name


internal["image"] = internal["image"].apply(clean_image_label)
external["image"] = external["image"].apply(clean_image_label)

internal["gate1"] = (internal["model"] == 'Other car').apply(int)
external["gate1"] = (external["model"] == 'Other car').apply(int)

le = LabelEncoder()
internal_filter = internal["model"] != 'Other car'
internal.loc[internal_filter, "gate2"] = le.fit_transform(
    internal.loc[internal_filter, "model"])
external_filter = external["model"] != 'Other car'
external.loc[external_filter, "gate2"] = le.transform(
    external.loc[external_filter, "model"])

le = LabelEncoder()
internal["color"] = le.fit_transform(internal["color"])
external["color"] = le.fit_transform(external["color"])

labels_to_save = ['image', 'gate1', 'gate2',
                  'color', 'lighting', 'source', 'model']
internal[labels_to_save].to_csv(
    base_dir / 'annotations/internal_new.csv', index=False)
external[labels_to_save].to_csv(
    base_dir / 'annotations/external_new.csv', index=False)
