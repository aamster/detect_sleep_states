import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import torch.utils.data


class Label(Enum):
    sleep = 0
    awake = 1
    onset = 2
    wakeup = 3


label_id_str_map = {
    0: 'sleep',
    1: 'awake',
    2: 'onset',
    3: 'wakeup'
}


class ClassifySegmentDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: Path,
        meta: pd.DataFrame,
        transform,
        sequence_length: int = 720,
        is_train: bool = True,
        limit_to_series_ids: Optional[List] = None
    ):
        """

        :param data_path:
            Path to series (parquet)
        :param meta:
            Table with columns series_id, start, end
            In train mode should also contain label, night
        :param sequence_length:
        :param is_train:
        :param transform:
        :param limit_to_series_ids
            Limit to loading these series ids
        """
        super().__init__()

        filters = [('series_id', 'in', limit_to_series_ids)] if (
                limit_to_series_ids is not None) else None
        series = pd.read_parquet(data_path, filters=filters)
        series = series.sort_values(['series_id', 'step'])

        self._series = series
        self._meta = meta
        self._is_train = is_train
        self._sequence_length = sequence_length
        self._transform = transform

    def __getitem__(self, index):
        row = self._meta.iloc[index]
        if not is_valid_sequence(
                seq_meta=row,
                sequence_length=self._sequence_length):
            raise ValueError(f'Invalid sequence: {row}')

        if self._is_train:
            if row['label'] in (Label.sleep.name, Label.awake.name):
                # Selecting a random sequence within [start,end]
                start = np.random.choice(
                    np.arange(row['start'],
                              row['end'] - self._sequence_length + 1),
                    size=1
                )[0]
            else:
                # Either Onset or wakeup transition
                # Selecting a random sequence that includes the transition
                # index
                start = np.random.choice(
                    np.arange(row['start'] - self._sequence_length,
                              row['start'] + 1),
                    size=1
                )[0]
        else:
            start = row['start']

        label = getattr(Label, row['label']).value \
            if 'label' in row else None

        series_data = self._series.loc[row.name]
        data = series_data.iloc[start:start+self._sequence_length].copy()

        data['timestamp'] = data['timestamp'].apply(
            lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S%z'))

        data = np.vstack([
            data['anglez'],
            data['enmo'],
            data['timestamp'].apply(lambda x: x.hour)
        ]).T

        data = self._transform(image=data)['image']

        data = {
            'sequence': data,
            'start': start,
            'end': start + self._sequence_length,
            'series_id': row.name
        }

        return data, label

    def __len__(self):
        return self._meta.shape[0]


def is_valid_sequence(seq_meta: pd.Series, sequence_length: int):
    is_valid = True
    if sequence_length > seq_meta['end'] - seq_meta['start']:
        if 'label' in seq_meta:
            if seq_meta['label'] in (Label.sleep.name, Label.awake.name):
                is_valid = False
        else:
            is_valid = False

    return is_valid
