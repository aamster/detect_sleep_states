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

ANGLEZ_MEAN = -8.810476
ANGLEZ_STD = 35.521877
ENMO_MEAN = 0.041315
ENMO_STD = 0.101829


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
        load_series: bool = True,
        event_width: int = 360,
        event_sigma: int = 10
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
            Number of consecutive timesteps to sample
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
        self._event_width = event_width
        self._event_sigma = event_sigma

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
            label = self.construct_target(
                events=self._events,
                series_id=row.name,
                start=start,
                sequence_length=self._sequence_length,
                event_width=self._event_width
            )
        else:
            label = None

        data = series_data.iloc[start:start+self._sequence_length].copy()

        data['timestamp'] = data['timestamp'].apply(
            lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S%z'))

        hour = data['timestamp'].apply(lambda x: x.hour)

        # https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        sequence = np.stack([
            (data['anglez'] - ANGLEZ_MEAN) / ANGLEZ_STD,
            (data['enmo'] - ENMO_MEAN) / ENMO_STD,
            hour_sin.values,
            hour_cos.values
        ])

        sequence_length = sequence.shape[1]

        if sequence_length < self._sequence_length:
            sequence = np.pad(
                sequence,
                pad_width=((0, 0), (0, self._sequence_length - sequence_length))
            )

        data = {
            'sequence': sequence,
            'start': start,
            'end': start + self._sequence_length,
            'sequence_length': sequence_length,
            'series_id': row.name
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

    @staticmethod
    def construct_target(
        events: pd.DataFrame,
        series_id: str,
        sequence_length: int,
        start: int,
        event_width: int = 360,
        event_sigma: int = 10
    ):
        events = events.loc[[series_id]]

        events = events[
            ((events['start'] >= start) &
             (events['start'] <= start + sequence_length)) |
            (events['start'] <= start) &
            (events['end'] >= start)]
        label = torch.zeros((sequence_length, 3),
                            dtype=torch.float)
        for event in events.itertuples():
            if event.event != Label.sleep.name:
                continue

            event_start = int(max(start, event.start))
            event_end = int(min(start + sequence_length, event.end))

            event_start = max(0, int(event_start - start))
            event_end = int(event_end - start)

            label[event_start:event_end+1, 0] = 1
            label[event_start, 1] = 1
            label[event_end, 2] = 1

        label[:, [1, 2]] = torch.tensor(
            _gaussian_label(
                label=label[:, [1, 2]].cpu().numpy(),
                offset=event_width,
                sigma=event_sigma
            ))

        return label


def is_valid_sequence(seq_meta: pd.Series, sequence_length: int):
    is_valid = True
    if sequence_length > seq_meta['end'] - seq_meta['start']:
        if 'label' in seq_meta:
            if seq_meta['label'] in ('sleep', 'awake'):
                is_valid = False
        else:
            is_valid = False

    return is_valid


# ref: https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/discussion/360236#2004730
def _gaussian_kernel(length: int, sigma: int = 3) -> np.ndarray:
    x = np.ogrid[-length : length + 1]
    h = np.exp(-(x**2) / (2 * sigma * sigma))  # type: ignore
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def _gaussian_label(label: np.ndarray, offset: int, sigma: int) -> np.ndarray:
    num_events = label.shape[1]
    for i in range(num_events):
        label[:, i] = np.convolve(label[:, i], _gaussian_kernel(offset, sigma), mode="same")

    return label
