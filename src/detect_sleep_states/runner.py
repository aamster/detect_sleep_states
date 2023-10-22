from collections import defaultdict
from typing import List, Dict, Optional

import albumentations
import argschema
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from lightning import Trainer
from tqdm import tqdm

from detect_sleep_states.classify_segment.data_module import SleepDataModule
from detect_sleep_states.classify_segment.dataset import Label
from detect_sleep_states.classify_segment.model import ClassifySegmentModel, \
    DetectSleepModel


class DetectSleepStatesSchema(argschema.ArgSchema):
    data_path = argschema.fields.InputFile(required=True)
    events_path = argschema.fields.InputFile(
        required=False,
        allow_none=True,
        default=None
    )
    batch_size = argschema.fields.Int(default=16)
    sequence_length = argschema.fields.Int(default=720)
    step_size = argschema.fields.Int(default=144)
    is_debug = argschema.fields.Bool(default=False)
    classify_segment_model_checkpoint = argschema.fields.InputFile(
        required=True)
    num_workers = argschema.fields.Int(default=0)
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
            limit_predict_batches=3 if self.args['is_debug'] else None
        )
        self._target = target

        if self.args['mode'] == 'validate':
            if self.args['events_path'] is None:
                raise ValueError('if validate, must provide events file')

    def run(self):
        preds = self._detect_sleep_segments()
        # if self.args['mode'] == 'validate':
        #     targets = self._get_sequence_targets(preds=preds)
        # else:
        #     targets = None
        # preds['label'] = targets
        return preds

    def _get_sequence_targets(self, preds: pd.DataFrame):
        events = self._clean_events()
        preds = preds.set_index('series_id')
        events = events.loc[preds.index.unique()]

        targets = []
        for series_id in preds.index.unique():
            series_preds = preds.loc[series_id]

            series_events = events.loc[series_id]
            series_events = series_events[
                (series_events['step'] >= series_preds['start'].min()) &
                (series_events['step'] <= series_preds['end'].max())
            ]
            series_targets = []

            events_idx = 0
            preds_idx = 0
            num_preds = 1 if isinstance(series_preds, pd.Series) else (
                series_preds.shape)[0]
            while (events_idx < series_events.shape[0] and
                   preds_idx < num_preds):
                event = series_events.iloc[events_idx]
                if isinstance(series_preds, pd.DataFrame):
                    pred = series_preds.iloc[preds_idx]
                else:
                    pred = series_preds

                event_idx = event['step']

                if pred['start'] <= event_idx <= pred['end']:
                    series_targets.append(event['event'])
                    events_idx += 1
                    preds_idx += 1
                else:
                    preds_idx += 1
            targets += series_targets
        return targets

    def _detect_sleep_segments(
        self
    ):
        series_meta = self._construct_valid_set()
        all_preds = []
        for series_id, meta in tqdm(series_meta.items(),
                                    desc='Series',
                                    total=len(series_meta)):
            self.logger.info(f'Series {series_id}')

            for meta_sequence in tqdm(meta, desc='Sequences'):
                data_mod = SleepDataModule(
                    batch_size=self.args['batch_size'],
                    data_path=self.args['data_path'],
                    meta=meta_sequence,
                    num_workers=self.args['num_workers'],
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
                for i in range(len(res)):
                    preds.append({
                        'series_id': series_id,
                        'start': meta_sequence.iloc[i]['start'],
                        'end': meta_sequence.iloc[i]['end'],
                        'pred': res[i]['pred']
                    })
                preds = pd.DataFrame(preds)
                preds = self._merge_repeated_preds(preds=preds)
                all_preds.append(preds)

                if self.args['is_debug']:
                    break

            if self.args['is_debug']:
                break
        all_preds = pd.concat(all_preds)
        all_preds = all_preds[all_preds['pred'].isin(
            [Label.onset.name, Label.wakeup.name])]
        return all_preds

    def _clean_events(self):
        events = pd.read_csv(self.args['events_path'])
        events = events[~events['step'].isna()]

        # Remove nights where there isn't a complete record
        event_counts = events.groupby(['series_id', 'night']).size()
        event_counts = event_counts.reset_index().rename(
            columns={0: 'count'})
        invalid = event_counts[event_counts['count'] < 2]
        invalid_series_nights = invalid['series_id'].str.cat(invalid['night'].astype(str), sep='_')
        series_nights = events['series_id'].str.cat(events['night'].astype(str), sep='_')
        events = events[~series_nights.isin(invalid_series_nights)]

        events = events.set_index('series_id')

        # protect against sequence going past the data
        events_ = []
        for series_id in events.index.unique():
            events_.append(events.loc[series_id].iloc[:-1])
        events = pd.concat(events_)
        return events

    def _construct_valid_set(self):
        events = self._clean_events()

        series_meta = defaultdict(list)
        for series_id in tqdm(events.index.unique()):
            series_events = events.loc[series_id]
            prev_night = 0
            i = 0
            start = 0
            end = -self.args['sequence_length']
            cur_night = series_events.iloc[i]['night']
            while i < len(series_events):
                rows = []
                while i < len(series_events) and (
                        cur_night == prev_night or cur_night == prev_night + 1):
                    end = series_events.iloc[i]['step']
                    prev_night = series_events.iloc[i]['night']
                    i += 1
                    cur_night = series_events.iloc[min(len(series_events)-1, i)]['night']

                for start_idx in range(int(start),
                                       int(end)+self.args['sequence_length'],
                                       self.args['step_size']):
                    end_idx = start_idx + self.args['sequence_length']

                    row = {
                        'series_id': series_id,
                        'start': start_idx,
                        'end': end_idx
                    }
                    rows.append(row)
                prev_night = cur_night
                start = (series_events.iloc[min(len(series_events)-1, i)]['step'] -
                         self.args['sequence_length'] +
                         self.args['step_size'])

                if len(rows) > 0:
                    meta = pd.DataFrame(rows)
                    series_meta[series_id].append(meta)
        return series_meta

    def _detect_sleep_segments_for_series(
            self,
            preds: List[Dict],
            target: Optional[pd.DataFrame] = None):
        if target is not None:
            target = target.set_index('series_id')
        all_series_preds = self.construct_series_ranges(
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
    def _merge_repeated_preds(preds: pd.DataFrame):
        """
        if pred is e.g. onset onset onset, then this will merge into a
        single sequence, and adjust the record to have start -> of the
        entire seq
        :return:
        """
        series_id = preds.iloc[0]['series_id']
        merged = []
        i = 0
        while i < len(preds):
            pred = preds.iloc[i]
            start = pred['start']
            end = pred['end']
            while i < len(preds) and preds.iloc[i]['pred'] == pred['pred']:
                end = pred['end']
                i += 1
            merged.append({
                'series_id': series_id,
                'start': start,
                'end': end,
                'pred': pred['pred']
            })
        merged = pd.DataFrame(merged)
        return merged

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
