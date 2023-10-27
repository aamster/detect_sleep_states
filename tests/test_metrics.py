import tempfile
from pathlib import Path

import pandas as pd
import pytest

from detect_sleep_states.classify_segment.metrics import calc_accuracy


def test_calc_accuracy_all_correct():
    preds = pd.DataFrame({
        'series_id': ['series1', 'series1'],
        'pred': ['onset', 'wakeup'],
        'start': [0, 200],
        'end': [100, 300]
    })

    events = pd.DataFrame({
        'series_id': ['series1', 'series1'],
        'night': [1, 1],
        'event': ['onset', 'wakeup'],
        'step': [50, 250]
    })

    res = \
        calc_accuracy(
            preds=preds,
            events=events)
    assert res['TP'] == 2
    assert res['FP'] == 0
    assert res['FN'] == 0


@pytest.mark.parametrize(
    'start, end, pred, expected', [
        pytest.param(
    [100, 200, 330, 400],
            [175, 325, 375, 475],
            ['wakeup', 'onset', 'wakeup', 'onset'],
            {
                'TP': 2,
                'FP': 2,
                'FN': 1
            }
        ),
        pytest.param(
    [50, 100, 300, 400],
            [75, 200, 400, 500],
            ['wakeup', 'onset', 'wakeup', 'onset'],
            {
                'TP': 3,
                'FP': 1,
                'FN': 0
            }
        ),
        pytest.param(
    [100, 300, 400, 1000],
            [200, 400, 500, 2000],
            ['onset', 'wakeup', 'onset', 'wakeup'],
            {
                'TP': 1 + 1 + 1,
                'FP': 1,
                'FN': 0
            }
        ),
        pytest.param(
    [100, 250, 300, 325, 425, 1000],
            [200, 300, 325, 400, 475, 2000],
            ['onset', 'wakeup', 'onset', 'wakeup', 'onset', 'wakeup'],
            {
                'TP': 1 + 1 + 1,
                'FP': 1 + 1 + 1,
                'FN': 0
            }
        )
    ]
)
def test_calc_accuracy_fp(start, end, pred, expected):
    preds = pd.DataFrame({
        'series_id': ['series1'] * len(pred),
        'pred': pred,
        'start': start,
        'end': end
    })

    events = pd.DataFrame({
        'series_id': ['series1', 'series1', 'series1'],
        'night': [1, 1, 1],
        'event': ['onset', 'wakeup', 'onset'],
        'step': [150, 350, 450]
    })
    res = \
        calc_accuracy(
            preds=preds,
            events=events)
    assert res['TP'] == expected['TP']
    assert res['FP'] == expected['FP']
    assert res['FN'] == expected['FN']


@pytest.mark.parametrize(
    'step, event, expected', [
        pytest.param(
            [75, 250, 400],
            ['onset', 'wakeup', 'onset'],
            {
                'TP': 1,
                'FP': 1,
                'FN': 2
            }
        ),
        pytest.param(
            [25, 75, 375],
            ['onset', 'wakeup', 'onset'],
            {
                'TP': 0,
                'FP': 2,
                'FN': 1 + 1 + 1
            }
        ),
        pytest.param(
            [75, 375, 1000],
            ['onset', 'wakeup', 'onset'],
            {
                'TP': 1 + 1,
                'FP': 0,
                'FN': 1
            }
        ),
        pytest.param(
            [75, 250, 300, 400, 1000],
            ['onset', 'wakeup', 'onset', 'wakeup', 'onset'],
            {
                'TP': 1 + 1,
                'FP': 0,
                'FN': 1 + 1 + 1
            }
        )
    ]
)
def test_calc_accuracy_fn(step, event, expected):
    preds = pd.DataFrame({
        'series_id': ['series1', 'series1'],
        'pred': ['onset', 'wakeup'],
        'start': [50, 350],
        'end': [100, 450]
    })

    events = pd.DataFrame({
        'series_id': ['series1'] * len(step),
        'night': [1] * len(step),
        'event': event,
        'step': step
    })
    res = \
        calc_accuracy(
            preds=preds,
            events=events)
    assert res['TP'] == expected['TP']
    assert res['FP'] == expected['FP']
    assert res['FN'] == expected['FN']


def test_calc_accuracy_all_errors():
    preds = pd.DataFrame({
        'series_id': ['series1'],
        'pred': ['onset'],
        'start': [0],
        'end': [100]
    })

    events = pd.DataFrame({
        'series_id': ['series1', 'series1', 'series1'],
        'night': [1, 1, 1],
        'event': ['onset', 'wakeup', 'onset'],
        'step': [150, 251, 300]
    })

    res = \
        calc_accuracy(
            preds=preds,
            events=events)
    assert res['TP'] == 0
    assert res['FP'] == 1
    assert res['FN'] == 3


def test_calc_accuracy_multiple_pred_matches():
    preds = pd.DataFrame({
        'series_id': ['series1'],
        'pred': ['onset'],
        'start': [0],
        'end': [100]
    })

    events = pd.DataFrame({
        'series_id': ['series1', 'series1'],
        'night': [1, 1],
        'event': ['onset', 'wakeup'],
        'step': [50, 75]
    })
    res = \
        calc_accuracy(
            preds=preds,
            events=events)
    assert res['TP'] == 1
    assert res['FP'] == 0
    assert res['FN'] == 1


def test_calc_accuracy_multiple_event_matches():
    preds = pd.DataFrame({
        'series_id': ['series1', 'series1', 'series1'],
        'pred': ['onset', 'wakeup', 'onset'],
        'start': [25, 55, 65],
        'end': [70, 125, 200]
    })

    events = pd.DataFrame({
        'series_id': ['series1', 'series1'],
        'night': [1, 1],
        'event': ['onset', 'wakeup'],
        'step': [50, 75]
    })
    res = \
        calc_accuracy(
            preds=preds,
            events=events)
    assert res['TP'] == 2
    assert res['FP'] == 1
    assert res['FN'] == 0
