import random
from pathlib import Path
from typing import Optional

import lightning
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from detect_sleep_states.dataset import \
    ClassifySegmentDataset, is_valid_sequence, Label
from detect_sleep_states.util import clean_events


class SleepDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        meta: pd.DataFrame,
        data_path: Path,
        events_path: Optional[Path] = None,
        sequence_length: int = 720,
        train_transform: Optional[transforms.Compose] = None,
        inference_transform: Optional[transforms.Compose] = None,
        is_debug: bool = False,
        load_series: bool = True
    ):
        super().__init__()

        meta = meta[meta.apply(lambda x: is_valid_sequence(
            seq_meta=x, sequence_length=sequence_length), axis=1)]

        if meta.index.name != 'series_id':
            meta = meta.set_index('series_id')
        self._meta = meta
        self._events = clean_events(events_path=str(events_path))
        self._series_ids = self._meta.index.unique()
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._data_path = data_path
        self._train = None
        self._val = None
        self._predict = None
        self._sequence_length = sequence_length
        self._train_transform = train_transform
        self._inference_transform = inference_transform
        self._is_debug = is_debug
        self._load_series = load_series

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            idxs = np.arange(len(self._series_ids))
            rng = np.random.default_rng(12345)
            rng.shuffle(idxs)
            train_idxs = idxs[:int(len(idxs) * .7)]
            val_idxs = idxs[int(len(idxs) * .7):]

            train_series_ids = self._series_ids[train_idxs]
            val_series_ids = self._series_ids[val_idxs]
            if self._is_debug:
                train_series_id = pd.Series(self._meta.index).sample(1)
                meta = self._meta.loc[train_series_id]

                train = pd.concat([
                    meta.loc[(meta['night'] != meta['night'].max()) & (meta['label'] == 'sleep')].sample(1),
                    meta.loc[(meta['night'] != meta['night'].max()) & (
                                meta['label'] == 'awake')].sample(1),
                    meta.loc[(meta['night'] != meta['night'].max()) & (
                            meta['label'] == 'onset')].sample(1),
                    meta.loc[(meta['night'] != meta['night'].max()) & (
                            meta['label'] == 'wakeup')].sample(1)
                ])

                rng = np.random.default_rng(1234)
                onset_sequence_start = rng.choice(range(int(self._sequence_length / 2)))
                wakeup_sequence_start = rng.choice(
                    range(int(self._sequence_length / 2)))
                sleep_sequence_start = rng.choice(
                    range(
                        train.loc[train['label'] == 'sleep', 'start'].iloc[0],
                        train.loc[train['label'] == 'sleep', 'end'].iloc[0] - self._sequence_length
                    )
                )
                awake_sequence_start = rng.choice(
                    range(
                        train.loc[train['label'] == 'awake', 'start'].iloc[0],
                        train.loc[train['label'] == 'awake', 'end'].iloc[0] - self._sequence_length
                    )
                )
                train.loc[train['label'] == 'sleep', 'start'] = sleep_sequence_start
                train.loc[train['label'] == 'sleep', 'end'] =  sleep_sequence_start + self._sequence_length

                train.loc[train['label'] == 'awake', 'start'] = awake_sequence_start
                train.loc[train['label'] == 'awake', 'end'] =  awake_sequence_start + self._sequence_length

                train.loc[train['label'] == 'onset', 'start'] -= onset_sequence_start
                train.loc[train['label'] == 'onset', 'end'] = train.loc[train['label'] == 'onset', 'start'] + self._sequence_length

                train.loc[train['label'] == 'wakeup', 'start'] -= wakeup_sequence_start
                train.loc[train['label'] == 'wakeup', 'end'] = train.loc[train['label'] == 'wakeup', 'start'] + self._sequence_length
                train_series_ids = train.index.unique()
            else:
                train = self._meta.loc[train_series_ids]
            self._train = ClassifySegmentDataset(
                data_path=self._data_path,
                sequences=train,
                sequence_length=self._sequence_length,
                events=self._events.loc[train_series_ids],
                is_train=False if self._is_debug else True,
                transform=self._train_transform,
                limit_to_series_ids=train_series_ids
            )

            self._val = ClassifySegmentDataset(
                data_path=self._data_path,
                sequences=self.get_test_set(meta=self._meta.loc[val_series_ids]),
                events=self._events.loc[val_series_ids],
                sequence_length=self._sequence_length,
                is_train=False,
                transform=self._inference_transform,
                limit_to_series_ids=val_series_ids,
            )
        elif stage == 'predict':
            self._predict = ClassifySegmentDataset(
                data_path=self._data_path,
                sequences=self._meta,
                sequence_length=self._sequence_length,
                is_train=False,
                transform=self._inference_transform,
                limit_to_series_ids=self._series_ids,
                load_series=self._load_series
            )

    def get_test_set(self, meta: pd.DataFrame):
        data = []
        for row in meta.itertuples(index=True):
            if 'night' in meta:
                last_night = meta.loc[row.Index]['night'].max()
            else:
                last_night = None

            if (getattr(row, 'label', None) is not None and
                    row.label in (Label.sleep.name, Label.awake.name)):
                # limit the end to not go beyond the sequence
                end = row.end - self._sequence_length \
                    if row.night == last_night and not self._is_debug else row.end
            elif getattr(row, 'label', None) is None:
                end = row.end
            else:
                continue
            starts = np.arange(
                row.start,
                end,
                self._sequence_length)
            for start in starts:
                datum = {
                    'series_id': row.Index,
                    'start': start,
                    'end': start + self._sequence_length
                }
                if getattr(row, 'label', None) is not None:
                    if datum['end'] > row.end:
                        if row.label == Label.sleep.name:
                            label = 'wakeup'
                        else:
                            label = 'onset'
                    else:
                        label = row.label
                    datum['label'] = label
                if getattr(row, 'night', None) is not None:
                    datum['night'] = row.night
                data.append(datum)
        data = pd.DataFrame(data)
        data = data.set_index('series_id')

        if self._is_debug and 'label' in data:
            data = pd.concat([
                data[data['label'] == 'sleep'].sample(1),
                data[data['label'] == 'awake'].sample(1),
                data[data['label'] == 'onset'].sample(1),
                data[data['label'] == 'wakeup'].sample(1)
            ])
        return data

    def train_dataloader(self):
        return DataLoader(
            self._train,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self._val,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=False
        )

    def predict_dataloader(self):
        return DataLoader(
            self._predict,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=False
        )

    @property
    def predict(self) -> ClassifySegmentDataset:
        return self._predict
