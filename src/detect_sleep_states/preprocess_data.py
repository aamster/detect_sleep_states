from pathlib import Path

import pandas as pd

from detect_sleep_states.util import clean_events


def generate_training_meta(events_path: Path):
    events = clean_events(events_path=str(events_path))
    events['step'] = events['step'].astype(int)
    events_full = []

    for series_id in events.index.unique():
        series_events = events.loc[series_id]
        prev_event_step = 0
        prev_night = 0
        for i in range(series_events.shape[0]):
            events_full.append({
                'series_id': series_id,
                'night': series_events.iloc[i]['night'],
                'label': series_events.iloc[i]['event'],
                'start': series_events.iloc[i]['step'],
                'end': series_events.iloc[i]['step']
            })
            if series_events.iloc[i]['event'] == 'onset':
                prev_event = 'awake'
            else:
                prev_event = 'sleep'
            event = {
                'series_id': series_id,
                'label': prev_event,
                'start': prev_event_step,
                'end': series_events.iloc[i]['step']
            }

            if (series_events.iloc[i]['night'] == prev_night or
                    series_events.iloc[i]['night'] == prev_night + 1):
                events_full.append(event)

            prev_night = series_events.iloc[i]['night']
            prev_event_step = series_events.iloc[i]['step']

    events_full = pd.DataFrame(events_full)
    events_full = events_full.set_index('series_id')
    events_full = events_full.sort_values(['series_id', 'start'])
    return events_full
