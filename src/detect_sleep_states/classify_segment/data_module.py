from pathlib import Path
from typing import Optional

import lightning
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from detect_sleep_states.classify_segment.dataset import \
    ClassifySegmentDataset, is_valid_sequence, Label


class SleepDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        meta_path: Path,
        data_path: Path,
        sequence_length: int = 720,
        train_transform: Optional[transforms.Compose] = None,
        inference_transform: Optional[transforms.Compose] = None,
        is_debug: bool = False
    ):
        super().__init__()
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._meta_path = meta_path
        self._data_path = data_path
        self._train = None
        self._val = None
        self._sequence_length = sequence_length
        self._train_transform = train_transform
        self._inference_transform = inference_transform
        self._is_debug = is_debug

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            meta = pd.read_csv(self._meta_path)

            meta = meta[meta.apply(lambda x: is_valid_sequence(
                seq_meta=x, sequence_length=self._sequence_length), axis=1)]

            meta = meta.set_index('series_id')
            series_ids = meta.index.unique()
            idxs = np.arange(len(series_ids))
            rng = np.random.default_rng(12345)
            rng.shuffle(idxs)
            train_idxs = idxs[:int(len(idxs) * .7)]
            val_idxs = idxs[int(len(idxs) * .7):]

            train_series_ids = series_ids[train_idxs]
            val_series_ids = series_ids[val_idxs]
            if self._is_debug:
                meta = pd.concat([
                    meta[meta['label'] == 'sleep'].sample(1),
                    meta[meta['label'] == 'awake'].sample(1)
                ])
                train_series_ids = meta.index.unique()
                train = self._get_test_set(meta=meta)
            else:
                train = meta.loc[train_series_ids]
            self._train = ClassifySegmentDataset(
                data_path=self._data_path,
                meta=train,
                sequence_length=self._sequence_length,
                is_train=False if self._is_debug else True,
                transform=self._train_transform,
                limit_to_series_ids=train_series_ids
            )

            self._val = ClassifySegmentDataset(
                data_path=self._data_path,
                meta=self._get_test_set(meta=meta.loc[val_series_ids]),
                sequence_length=self._sequence_length,
                is_train=False,
                transform=self._inference_transform,
                limit_to_series_ids=val_series_ids,
            ) if not self._is_debug else None

    def _get_test_set(self, meta: pd.DataFrame):
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
                    if row.night == last_night else row.end
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
                            label = Label.wakeup.name
                        else:
                            label = Label.onset.name
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
            self._val,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=False
        )
