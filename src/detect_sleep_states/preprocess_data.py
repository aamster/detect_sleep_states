from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm


def generate_training_meta(
    events: pd.DataFrame,
    sequence_length: int
):
    sequences = []
    for series_id in events.index.unique():
        series_events = events.loc[[series_id]]
        onset_wakeup_events = series_events[series_events['event'].isin(['onset', 'wakeup'])]
        for seq_start in range(0, int(series_events['end'].max()), sequence_length):
            contains_an_event = False
            seq_end = seq_start+sequence_length
            for event in onset_wakeup_events.itertuples():
                if seq_start <= event.start <= seq_end:
                    contains_an_event = True

            if contains_an_event:
                sequences.append({
                    'series_id': series_id,
                    'start': seq_start,
                    'end': seq_end
                })
    sequences = pd.DataFrame(sequences)
    sequences = sequences.set_index('series_id')
    sequences = sequences.sort_values(['series_id', 'start'])
    return sequences


def _get_missing_sequences(
    events: pd.DataFrame,
    series_lengths: Dict[str, int]
):
    missing = []

    for series_id in tqdm(events.index.unique(),
                          desc='finding missing sequences'):
        series_events = events.loc[series_id]
        i = 0
        while i < series_events.shape[0]:
            if pd.isna(series_events.iloc[i]['step']):
                if i == 0:
                    missing_start = 0
                else:
                    missing_start = series_events.iloc[i-1]['step'] + 1
                while (i < series_events.shape[0] and
                       pd.isna(series_events.iloc[i]['step'])):
                    i += 1

                if (i == series_events.shape[0] and
                    pd.isna(series_events.iloc[-1]['step'])):
                    # reached the end and it's still missing
                    # Many times the series extended past the number of
                    # "annotated" missing events. Truncate to the estimated
                    # number of missing timesteps based on the length of a day
                    num_timesteps_in_day = int(60*60*24/5)
                    last_annotated_day = series_events[~series_events['step'].isna()]['night'].max()
                    if not np.isnan(last_annotated_day):
                        num_expected_days = series_events['night'].max()
                        num_expected_timesteps = num_expected_days * num_timesteps_in_day
                        missing_end = min(num_expected_timesteps,
                                          series_lengths[series_id])
                    else:
                        # entire thing is missing
                        missing_end = series_lengths[series_id]

                else:
                    missing_end = series_events.iloc[i]['step'] - 1

                missing.append({
                    'series_id': series_id,
                    'start': missing_start,
                    'end': missing_end
                })
            else:
                i += 1

    missing = pd.DataFrame(missing)
    missing = missing.set_index('series_id')
    missing = missing.sort_values(['series_id', 'start'])
    return missing


def get_full_events(
    events: pd.DataFrame,
    series_lengths: Dict[str, int]
):
    if events.index.name != 'series_id':
        events = events.set_index('series_id')
    events = events.sort_index()

    missing = _get_missing_sequences(
        events=events,
        series_lengths=series_lengths
    )
    missing['event'] = 'missing'

    full = []
    for series_id in events.index.unique():
        series_events = events.loc[series_id]
        series_full = []
        prev_event_idx = -1
        for event_idx, event in enumerate(series_events.itertuples()):
            if not pd.isna(event.step):
                if event.event == 'onset':
                    prev_event = 'awake'
                else:
                    prev_event = 'sleep'
                series_full.append({
                    'series_id': series_id,
                    'start': prev_event_idx + 1,
                    'end': event.step - 1,
                    'event': prev_event
                })
                series_full.append({
                    'series_id': series_id,
                    'start': event.step,
                    'end': event.step,
                    'event': event.event
                })
                prev_event_idx = event.step

            if event_idx == series_events.shape[0] - 1:
                if event.event == 'onset':
                    next_event = 'sleep'
                else:
                    next_event = 'awake'

                series_full.append({
                    'series_id': series_id,
                    'start': event.step + 1,
                    'end': series_lengths[series_id],
                    'event': next_event
                })
        if series_id in missing.index.unique():
            if isinstance(missing.loc[series_id], pd.DataFrame):
                series_missing = set(missing.loc[series_id].apply(
                    lambda x: (x['start'], x['end']), axis=1).values)
            else:
                series_missing = [(
                    missing.loc[series_id]['start'],
                    missing.loc[series_id]['end']
                )]
        else:
            series_missing = []
        series_full = [x for x in series_full
                       if (x['start'], x['end']) not in series_missing]
        series_full = pd.DataFrame(
            series_full
        )
        if not series_full.empty:
            series_full = series_full.set_index('series_id')
            full.append(series_full)
    full = pd.concat(full)
    full = pd.concat([full, missing])
    full = full.sort_values(['series_id', 'start'])
    return full


def get_series_lengths(
    events: pd.DataFrame,
    series_path: Path
) -> Dict[str, int]:
    if events.index.name != 'series_id':
        events = events.set_index('series_id')
    series_lengths = {}
    for series_id in tqdm(events.index.unique(),
                          desc='Getting series lengths'):
        series_len = pd.read_parquet(
            series_path,
            filters=[('series_id', '=', series_id)],
            columns=[]
        ).shape[0]
        series_lengths[series_id] = series_len
    return series_lengths
