from typing import Union

import pandas as pd

from detect_sleep_states.util import clean_events


def calc_accuracy(
    preds: pd.DataFrame,
    events: Union[pd.DataFrame, str]
):
    if not isinstance(events, pd.DataFrame):
        events = clean_events(events_path=events)
    if events.index.name != 'series_id':
        events = events.set_index('series_id')

    if preds.index.name != 'series_id':
        preds = preds.set_index('series_id')

    TP = 0
    FP = 0
    FN = 0

    for series_id in events.index.unique():
        series_events = events.loc[[series_id]]
        series_preds = preds.loc[[series_id]]

        event_matches = dict()
        for event_idx in range(series_events.shape[0]):
            event_matches[event_idx] = set()

        for pred_idx, pred in enumerate(series_preds.itertuples()):
            for event_idx, event in enumerate(series_events.itertuples()):
                if pred.start <= event.step <= pred.end:
                    if pred.pred == event.event:
                        event_matches[event_idx].add(pred_idx)

        pred_matches = dict()
        for pred_idx in range(series_preds.shape[0]):
            pred_matches[pred_idx] = set()

        for event_idx, event in enumerate(series_events.itertuples()):
            for pred_idx, pred in enumerate(series_preds.itertuples()):
                if pred.start <= event.step <= pred.end:
                    if pred.pred == event.event:
                        pred_matches[pred_idx].add(event_idx)

        for pred_idx, event_idx_matches in pred_matches.items():
            TP += len(event_idx_matches)
            if len(event_idx_matches) == 0:
                FP += 1

        for event_idx, pred_idx_matches in event_matches.items():
            if len(pred_idx_matches) == 0:
                FN += 1

    p = TP / (TP + FP)
    r = TP / (TP + FN)

    if p + r == 0:
        f1 = 0.0
    else:
        f1 = 2 * p * r / (p + r)

    return {
        'precision': p,
        'recall': r,
        'f1': f1,
        'TP': TP,
        'FP': FP,
        'FN': FN
    }
