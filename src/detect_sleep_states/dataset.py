import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import torch.utils.data


class Label(Enum):
    status_quo = 0
    onset = 1
    wakeup = 2


label_id_str_map = {
    0: 'status_quo',
    1: 'onset',
    2: 'wakeup'
}


class ClassifySegmentDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: Path,
        sequences: pd.DataFrame,
        transform,
        sequence_length: int = 720,
        is_train: bool = True,
        events: Optional[pd.DataFrame] = None,
        limit_to_series_ids: Optional[List] = None,
        load_series: bool = True
    ):
        """

        :param data_path:
            Path to series (parquet)
        :param sequences:
            Table with columns series_id, start, end
            In train mode should also contain label, night
        :param events
            Table with all onset, wakeup events.
            Only relevant in training
        :param sequence_length:
        :param is_train:
        :param transform:
        :param limit_to_series_ids
            Limit to loading these series ids
        :param load_series
            Whether to load series. It takes a while, skip if not needed
        """
        super().__init__()

        filters = [('series_id', 'in', limit_to_series_ids)] if (
                limit_to_series_ids is not None) else None

        if load_series:
            series = pd.read_parquet(data_path, filters=filters)
            series = series.sort_values(['series_id', 'step'])
        else:
            series = None

        if events.index.name != 'series_id':
            events = events.set_index('series_id')

        self._series = series.set_index('series_id')
        self._sequences = sequences
        self._events = events
        self._is_train = is_train
        self._sequence_length = sequence_length
        self._transform = transform

    def __getitem__(self, index):
        row = self._sequences.iloc[index]
        if self._is_train:
            # shifting randomly between -4 hours and +4 hours
            start = row['start'] + np.random.randint(-int(60*60*4/5),
                                                     int(60*60*4/5))
            start = max(0, start)

        else:
            start = row['start']

        if self._events is not None:
            events = self._events.loc[row.name]
            events = events[(events['step'] >= start) &
                            (events['step'] <= start+self._sequence_length)]
            label = torch.zeros((self._sequence_length, 3), dtype=torch.long)
            label[:, 0] = 1
            for event in events.itertuples():
                label[
                    max(0, int(event.step - start)-360):
                    int(event.step - start)+360,
                    getattr(Label, event.event).value] = 1
                label[max(0, int(event.step - start)-360):
                      int(event.step - start)+360, 0] = 0

        else:
            label = None

        series_data = self._series.loc[row.name]
        data = series_data.iloc[start:start+self._sequence_length].copy()

        data['timestamp'] = data['timestamp'].apply(
            lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S%z'))

        sequence = np.stack([
            data['anglez'],
            data['enmo']
        ])

        if sequence.shape[1] < self._sequence_length:
            sequence = np.pad(
                sequence,
                pad_width=((0, 0), (0, self._sequence_length - sequence.shape[1]))
            )

        sequence = self._transform(image=sequence)['image']

        sequence = sequence.squeeze(dim=0)

        data = {
            'sequence': sequence,
            'start': start,
            'end': start + self._sequence_length,
            'series_id': row.name,
            'hour': data['timestamp'].dt.hour.values
        }

        if label is None:
            return data
        else:
            return data, label

    def __len__(self):
        return self._sequences.shape[0]

    @property
    def meta(self):
        return self._sequences


def is_valid_sequence(seq_meta: pd.Series, sequence_length: int):
    is_valid = True
    if sequence_length > seq_meta['end'] - seq_meta['start']:
        if 'label' in seq_meta:
            if seq_meta['label'] in ('sleep', 'awake'):
                is_valid = False
        else:
            is_valid = False

    return is_valid
