from collections import defaultdict
from typing import Union

import pandas as pd
import torch
import torchmetrics

from detect_sleep_states.util import clean_events


def calc_accuracy(
    preds: pd.DataFrame,
    events: Union[pd.DataFrame, str]
):
    label_idx_map = {
        'no_event': 0,
        'onset': 1,
        'wakeup': 2
    }

    all_preds = []
    all_events = []

    if not isinstance(events, pd.DataFrame):
        events = clean_events(events_path=events)
    if events.index.name != 'series_id':
        events = events.set_index('series_id')

    if preds.index.name != 'series_id':
        preds = preds.set_index('series_id')
    for series_id in events.index.unique():
        series_events = events.loc[[series_id]]
        series_preds = preds.loc[[series_id]]
        series_preds_corrected = series_preds.to_dict(orient='records')
        series_events_corrected = series_events.to_dict(orient='records')

        event_matches = dict()
        for event_idx in range(series_events.shape[0]):
            event_matches[event_idx] = set()

        for pred_idx, pred in enumerate(series_preds.itertuples()):
            for event_idx, event in enumerate(series_events.itertuples()):
                if pred.start <= event.step <= pred.end:
                    event_matches[event_idx].add(pred_idx)

        pred_matches = dict()
        for pred_idx in range(series_preds.shape[0]):
            pred_matches[pred_idx] = set()

        for event_idx, event in enumerate(series_events.itertuples()):
            for pred_idx, pred in enumerate(series_preds.itertuples()):
                if pred.start <= event.step <= pred.end:
                    pred_matches[pred_idx].add(event_idx)

        for pred_idx, event_idx_matches in pred_matches.items():
            if len(event_idx_matches) == 0:
                pred = series_preds.iloc[pred_idx]
                to_insert = {
                    'series_id': series_id,
                    'event': 'no_event',
                    'step': None
                }
                i = 0
                inserted = False
                while i < len(series_events_corrected):
                    if series_events_corrected[i]['event'] == 'no_event':
                        while (i < len(series_events_corrected) and
                               series_events_corrected[i]['event'] == 'no_event'):
                            i += 1
                    if i == len(series_events_corrected):
                        if series_events_corrected[-1]['event'] == 'no_event':
                            series_events_corrected.append(to_insert)
                            inserted = True
                            break
                    else:
                        if pred['end'] <= series_events_corrected[i]['step']:
                            series_events_corrected.insert(i, to_insert)
                            inserted = True
                            break
                    i += 1

                if not inserted:
                    series_events_corrected.append(to_insert)

        for event_idx, pred_idx_matches in event_matches.items():
            if len(pred_idx_matches) == 0:
                event = series_events.iloc[event_idx]
                to_insert = {
                    'series_id': series_id,
                    'pred': 'no_event',
                    'start': None,
                    'end': None
                }
                i = 0
                inserted = False
                while i < len(series_preds_corrected):
                    if series_preds_corrected[i]['pred'] == 'no_event':
                        while (i < len(series_preds_corrected) and
                               series_preds_corrected[i]['pred'] == 'no_event'):
                            i += 1
                    if i == len(series_preds_corrected):
                        if series_preds_corrected[-1]['pred'] == 'no_event':
                            series_preds_corrected.append(to_insert)
                            inserted = True
                            break
                    else:
                        if event['step'] <= series_preds_corrected[i]['start']:
                            series_preds_corrected.insert(i, to_insert)
                            inserted = True
                            break
                    i += 1

                if not inserted:
                    series_preds_corrected.append(to_insert)

        all_preds += [label_idx_map[x['pred']] for x in series_preds_corrected]
        all_events += [label_idx_map[x['event']] for x in series_events_corrected]

    precision = torchmetrics.classification.MulticlassPrecision(
        num_classes=3)
    recall = torchmetrics.classification.MulticlassRecall(
        num_classes=3
    )
    f1 = torchmetrics.classification.MulticlassF1Score(
        num_classes=3
    )
    precision.update(
        preds=torch.tensor(all_preds),
        target=torch.tensor(all_events)
    )
    recall.update(
        preds=torch.tensor(all_preds),
        target=torch.tensor(all_events)
    )
    f1.update(
        preds=torch.tensor(all_preds),
        target=torch.tensor(all_events)
    )
    return (precision.compute(), recall.compute(), f1.compute(),
            all_preds, all_events)
