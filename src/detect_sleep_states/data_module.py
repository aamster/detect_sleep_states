import random
from pathlib import Path
from typing import Optional, List, Tuple

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

        if meta.index.name != 'series_id':
            meta = meta.set_index('series_id')
        self._meta = meta
        self._events = clean_events(events_path=str(events_path)) \
            if events_path is not None else None
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

    @staticmethod
    def get_train_val_split(series_ids: np.ndarray) -> (
            Tuple)[np.ndarray, np.ndarray]:
        idxs = np.arange(len(series_ids))
        rng = np.random.default_rng(12345)
        rng.shuffle(idxs)
        train_idxs = idxs[:int(len(idxs) * .7)]
        val_idxs = idxs[int(len(idxs) * .7):]

        train_series_ids = series_ids[train_idxs]
        val_series_ids = series_ids[val_idxs]

        return train_series_ids, val_series_ids

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            train_series_ids, val_series_ids = self.get_train_val_split(
                series_ids=self._series_ids)
            if self._is_debug:
                train_series_ids = pd.Series(self._meta.index).sample(
                    1, random_state=1234)
                train = self.get_dataset(meta=self._meta.loc[train_series_ids])
            else:
                train = self._meta.loc[train_series_ids]
            self._train = ClassifySegmentDataset(
                data_path=self._data_path,
                sequences=train,
                sequence_length=self._sequence_length,
                events=self._events.loc[train_series_ids],
                is_train=False if self._is_debug else True,
                transform=self._train_transform,
                limit_to_series_ids=train_series_ids.tolist()
            )

            self._val = ClassifySegmentDataset(
                data_path=self._data_path,
                sequences=self.get_dataset(meta=self._meta.loc[val_series_ids]),
                events=self._events.loc[val_series_ids],
                sequence_length=self._sequence_length,
                is_train=False,
                transform=self._inference_transform,
                limit_to_series_ids=val_series_ids.tolist(),
            )
        elif stage == 'predict':
            self._predict = ClassifySegmentDataset(
                data_path=self._data_path,
                events=(self._events.loc[self._series_ids]
                        if self._events is not None else None),
                sequences=self._meta,
                sequence_length=self._sequence_length,
                is_train=False,
                transform=self._inference_transform,
                limit_to_series_ids=self._series_ids.tolist(),
                load_series=self._load_series
            )

    def get_dataset(self, meta: pd.DataFrame):
        data = []
        for series_id in meta.index.unique():
            series_meta = meta.loc[series_id]
            series_events = self._events.loc[series_id]
            start = 0
            end = series_meta['end'].max()

            for seq_start in range(start, end, self._sequence_length):
                seq_end = seq_start + self._sequence_length
                if series_events[
                    (series_events['step'] >= seq_start) &
                    (series_events['step'] <= seq_end)
                ]['event'].nunique() == 2:
                    data.append({
                        'series_id': series_id,
                        'start': seq_start
                    })
        data = pd.DataFrame(data)
        data = data.set_index('series_id')

        if self._is_debug:
            data = data.sample(2, random_state=1234)
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

    @property
    def train(self) -> ClassifySegmentDataset:
        return self._train
