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
    missing = 2
    onset = 3
    wakeup = 4


label_id_str_map = {
    0: 'sleep',
    1: 'awake',
    2: 'missing',
    3: 'onset',
    4: 'wakeup'
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
        series_data = self._series.loc[row.name]

        if self._is_train:
            # shifting randomly between -2 hours and +2 hours
            start = row['start'] + np.random.randint(-int(60*60*2/5),
                                                     int(60*60*2/5))
            start = max(0, start)

            # preventing going past the end
            start = min(series_data.shape[0] - self._sequence_length,
                        start)

        else:
            start = row['start']

        if self._events is not None:
            events = self._events.loc[[row.name]]

            events = events[
                ((events['start'] >= start) &
                 (events['start'] <= start + self._sequence_length)) |
                (events['start'] <= start) &
                (events['end'] >= start)]
            label = torch.zeros((self._sequence_length, len(Label)),
                                dtype=torch.long)
            for event in events.itertuples():
                event_start = int(max(start, event.start))
                event_end = int(min(start + self._sequence_length, event.end))

                if event.event in (Label.onset.name, Label.wakeup.name):
                    event_start = max(0, int(event_start - start)-360)
                    event_end = int(event_end - start)+360
                else:
                    event_start = int(event_start - start)
                    event_end = int(event_end - start)

                label[event_start:event_end,
                      getattr(Label, event.event).value] = 1

            label[torch.where(label[:, Label.onset.value] == 1)[0], :3] = 0
            label[torch.where(label[:, Label.wakeup.value] == 1)[0], :3] = 0
        else:
            label = None

        data = series_data.iloc[start:start+self._sequence_length].copy()

        data['timestamp'] = data['timestamp'].apply(
            lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S%z'))

        sequence = np.stack([
            data['anglez'],
            data['enmo']
        ])

        sequence_length = sequence.shape[1]

        hour = data['timestamp'].apply(lambda x: x.hour).values
        if sequence_length < self._sequence_length:
            sequence = np.pad(
                sequence,
                pad_width=((0, 0), (0, self._sequence_length - sequence_length))
            )
            hour = np.pad(
                hour,
                pad_width=(0, self._sequence_length - data.shape[0]),
                constant_values=hour[-1]
            )

        sequence = self._transform(image=sequence)['image']

        sequence = sequence.squeeze(dim=0)

        data = {
            'sequence': sequence,
            'start': start,
            'end': start + self._sequence_length,
            'sequence_length': sequence_length,
            'series_id': row.name,
            'hour': hour
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
