import os
from collections import defaultdict
from typing import List, Dict, Optional

import albumentations
import argschema
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from lightning import Trainer

from detect_sleep_states.classify_segment.data_module import SleepDataModule
from detect_sleep_states.classify_segment.dataset import Label, \
    label_id_str_map
from detect_sleep_states.classify_segment.model import ClassifySegmentModel, \
    DetectSleepModel


class DetectSleepStatesSchema(argschema.ArgSchema):
    data_path = argschema.fields.InputFile(required=True)
    meta_path = argschema.fields.InputFile(required=True)
    batch_size = argschema.fields.Int(default=16)
    sequence_length = argschema.fields.Int(default=720)
    is_debug = argschema.fields.Bool(default=False)
    classify_segment_model_checkpoint = argschema.fields.InputFile(
        required=True)
    mode = argschema.fields.String()


class DetectSleepStatesRunner(argschema.ArgSchemaParser):
    default_schema = DetectSleepStatesSchema

    def __init__(
        self,
        target: Optional[pd.DataFrame] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self._model = ClassifySegmentModel.load_from_checkpoint(
            checkpoint_path=self.args['classify_segment_model_checkpoint'],
            map_location=torch.device('cpu') if not torch.has_cuda else None,
            model=DetectSleepModel(),
            learning_rate=1e-3,
            hyperparams={},
            batch_size=self.args['batch_size']
        )
        self._trainer = Trainer(
            limit_predict_batches=1 if self.args['is_debug'] else None
        )
        self._target = target

        meta = pd.read_csv(self.args['meta_path'])
        self._data_mod = SleepDataModule(
            batch_size=self.args['batch_size'],
            data_path=self.args['data_path'],
            meta=meta,
            num_workers=os.cpu_count(),
            train_transform=albumentations.Compose([
                albumentations.Normalize(mean=0, std=1),
                ToTensorV2()
            ]),
            inference_transform=albumentations.Compose([
                albumentations.Normalize(mean=0, std=1),
                ToTensorV2()
            ]),
            sequence_length=self.args['sequence_length'],
            load_series=False
        )
        self._data_mod.setup(stage='predict')

    def run(self):
        preds = self._detect_sleep_segments()
        if self.args['mode'] == 'validate':
            targets = self._get_sequence_targets(preds=preds)
        else:
            targets = None
        preds['label'] = targets
        return preds

    def _get_sequence_targets(self, preds: pd.DataFrame):
        events = self._data_mod.predict.meta[
            self._data_mod.predict.meta['label'].isin([Label.onset.name,
                                                       Label.wakeup.name])]
        preds = preds.set_index('series_id')

        targets = []
        for series_id in events.index.unique():
            series_events = events.loc[series_id]
            series_preds = preds.loc[series_id]

            events_idx = 0
            preds_idx = 0
            while (events_idx < series_events.shape[0] and
                   preds_idx < series_preds.shape[0]):
                event = series_events.iloc[events_idx]
                pred = series_preds.iloc[preds_idx]

                event_idx = event['start']  # start and end the same

                if pred['start'] <= event_idx <= pred['end']:
                    targets.append(event['label'])
                    events_idx += 1
                else:
                    targets.append('no_event')
                    preds_idx += 1
            if self.args['is_debug']:
                break
        return targets

    def _detect_sleep_segments(
        self,
        step_size: int = 144
    ):
        all_preds = []
        series_sequences = self.construct_series_sequences()
        for series_id, sequences in series_sequences.items():
            for sequence_idxs in sequences:
                meta = self._construct_predict_set(
                    series_id=series_id,
                    sequence_idxs=sequence_idxs,
                    step_size=step_size
                )
                data_mod = SleepDataModule(
                    batch_size=self.args['batch_size'],
                    data_path=self.args['data_path'],
                    meta=meta,
                    num_workers=os.cpu_count(),
                    train_transform=albumentations.Compose([
                        albumentations.Normalize(mean=0, std=1),
                        ToTensorV2()
                    ]),
                    inference_transform=albumentations.Compose([
                        albumentations.Normalize(mean=0, std=1),
                        ToTensorV2()
                    ]),
                    sequence_length=self.args['sequence_length']
                )
                data_mod.setup(stage='predict')

                res = self._trainer.predict(
                    model=self._model,
                    datamodule=data_mod
                )
                res = res[0]

                preds = []
                num_preds_to_average = (
                    int(self.args['sequence_length'] / step_size))
                for i in range(0, len(res)):
                    preds_to_average = torch.vstack([
                        res[i]['scores'] for i in
                        range(i, max(i-num_preds_to_average, -1), -1)])
                    scores_averaged = preds_to_average.mean(dim=0)
                    pred = torch.argmax(scores_averaged)
                    pred = label_id_str_map[pred.item()]
                    preds.append({
                        'series_id': series_id,
                        'start': meta.iloc[i]['start'],
                        'end': meta.iloc[i]['start'] + step_size,
                        'pred': pred
                    })
                if step_size < self.args['sequence_length']:
                    # fill in remaining sequences
                    for i in range(len(res) - num_preds_to_average + 1,
                                   len(res)):
                        preds_to_average = torch.vstack([
                            res[i]['scores'] for i in
                            range(i, len(res))])
                        scores_averaged = preds_to_average.mean(dim=0)
                        pred = torch.argmax(scores_averaged)
                        pred = label_id_str_map[pred.item()]
                        preds.append({
                            'series_id': series_id,
                            'start': meta.iloc[i]['end'] - step_size,
                            'end': meta.iloc[i]['end'],
                            'pred': pred
                        })
                preds = pd.DataFrame(preds)
                all_preds.append(preds)

                if self.args['is_debug']:
                    break

            if self.args['is_debug']:
                break
        all_preds = pd.concat(all_preds)
        return all_preds

    def _construct_predict_set(
        self,
        series_id: str,
        sequence_idxs: List[int],
        step_size: int
    ):
        meta = self._data_mod.predict.meta.iloc[sequence_idxs]
        rows = []
        for row in meta.itertuples():
            if row.label in (Label.onset.name, Label.wakeup.name):
                continue
            for start in range(row.start, row.end, step_size):
                end = start + self.args['sequence_length']
                if end < row.end:
                    label = row.label
                else:
                    if row.label == Label.sleep.name:
                        label = Label.wakeup.name
                    else:
                        label = Label.onset.name
                rows.append({'series_id': series_id, 'start': start,
                             'end': end, 'night': row.night,
                             'label': label})
        meta = pd.DataFrame(rows)
        return meta

    def construct_series_sequences(self):
        series_ends = dict()
        meta = self._data_mod.predict.meta
        series_sequences = defaultdict(list)
        for i, row in enumerate(meta.itertuples()):
            series_id = row.Index
            if series_id not in series_sequences:
                series_sequences[series_id].append([i])
            elif (meta.iloc[i]['start'] >
                  series_ends[series_id] + 1):
                series_sequences[series_id].append([i])
            else:
                series_sequences[series_id][-1].append(i)
            series_ends[series_id] = max(
                series_ends.get(series_id, -float('inf')),
                row.end
            )
        return series_sequences

    def _detect_sleep_segments_for_series(
            self,
            preds: List[Dict],
            target: Optional[pd.DataFrame] = None):
        if target is not None:
            target = target.set_index('series_id')
        all_series_preds = self.construct_series_sequences(
            preds=preds, target=target
        )

        res = []
        target = self._merge_repeated_preds(
            all_series_preds=all_series_preds,
            target=target
        )
        self.fix(all_series_preds=all_series_preds)

        for series_id, series_preds in all_series_preds.items():
            for sequence in series_preds:
                for pred in sequence:
                    res.append(pred[0])
        return res, target

    @staticmethod
    def _merge_repeated_preds(all_series_preds: Dict, target: pd.DataFrame):
        """
        if pred is e.g. onset onset onset, then this will merge into a
        single sequence, and adjust the record to have start -> of the
        entire seq
        :return:
        """
        targets = []

        for series_id, series_preds in all_series_preds.items():
            for seq_num, sequence in enumerate(series_preds):
                remove_idx = set()
                i = 0
                while i < len(sequence):
                    label = sequence[i][0]['pred']

                    if label in (Label.onset.name, Label.wakeup.name):
                        merged = target.iloc[sequence[i][2]].copy()
                        start_idx = i
                        i += 1
                        while (i < len(sequence) and
                               sequence[i][0]['pred'] == label):
                            remove_idx.add(i)
                            i += 1
                        merged['end'] = target.iloc[sequence[min(i, len(sequence)-1)][2] - 1][
                            'end']
                        index_to_update = sequence[start_idx][2]
                        if label in target.iloc[sequence[start_idx][2]:sequence[min(i, len(sequence)-1)][2]]['label'].tolist():
                            merged['label'] = label
                        target.iloc[index_to_update] = merged
                    else:
                        i += 1
                all_series_preds[series_id][seq_num] = [
                    sequence[i]
                    for i in range(len(sequence)) if i not in remove_idx]
                targets.append(
                    target.iloc[
                        [sequence[i][2] for i in range(len(sequence))
                         if i not in remove_idx]])
        target = pd.concat(targets)
        return target

    @staticmethod
    def fix(all_series_preds):
        for series_id, series_preds in all_series_preds.items():
            for seq_num, sequence in enumerate(series_preds):
                for i in range(1, len(sequence) - 1):
                    prev = sequence[i-1][0]['pred']
                    cur = sequence[i][0]['pred']
                    next = sequence[i + 1][0]['pred']

                    if (prev == Label.sleep.name and
                        next == Label.sleep.name):
                        all_series_preds[series_id][seq_num][i][0]['pred'] = (
                            Label.sleep.name)

                    if (prev == Label.awake.name and
                        next == Label.awake.name):
                        all_series_preds[series_id][seq_num][i][0]['pred'] = (
                            Label.awake.name)

                    if (prev == Label.awake.name and
                        next == Label.sleep.name):
                        all_series_preds[series_id][seq_num][i][0]['pred'] = (
                            Label.onset.name)

                    if (prev == Label.sleep.name and
                        next == Label.awake.name):
                        all_series_preds[series_id][seq_num][i][0]['pred'] = (
                            Label.wakeup.name)
