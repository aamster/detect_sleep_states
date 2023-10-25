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

    p, r, f1, preds, events = \
        calc_accuracy(
            preds=preds,
            events=events)
    assert preds == [1, 2]
    assert events == [1, 2]


@pytest.mark.parametrize(
    'start, end, pred, expected_preds, expected_events', [
        pytest.param(
    [100, 200, 330, 400],
            [175, 325, 375, 475],
            ['wakeup', 'onset', 'wakeup', 'onset'],
            [2, 1, 2, 1],
            [1, 0, 2, 1]
        ),
        pytest.param(
    [50, 100, 300, 400],
            [75, 200, 400, 500],
            ['wakeup', 'onset', 'wakeup', 'onset'],
            [2, 1, 2, 1],
            [0, 1, 2, 1]
        ),
        pytest.param(
    [100, 300, 400, 1000],
            [200, 400, 500, 2000],
            ['onset', 'wakeup', 'onset', 'wakeup'],
            [1, 2, 1, 2],
            [1, 2, 1, 0]
        ),
        pytest.param(
    [100, 250, 300, 325, 425, 1000],
            [200, 300, 325, 400, 475, 2000],
            ['onset', 'wakeup', 'onset', 'wakeup', 'onset', 'wakeup'],
            [1, 2, 1, 2, 1, 2],
            [1, 0, 0, 2, 1, 0]
        )
    ]
)
def test_calc_accuracy_fp(start, end, pred, expected_preds, expected_events):
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
    p, r, f1, preds, events = \
        calc_accuracy(
            preds=preds,
            events=events)
    assert preds == expected_preds
    assert events == expected_events


@pytest.mark.parametrize(
    'step, event, expected_preds, expected_events', [
        pytest.param(
            [75, 250, 400],
            ['onset', 'wakeup', 'onset'],
            [1, 0, 2],
            [1, 2, 1]
        ),
        pytest.param(
            [25, 75, 375],
            ['onset', 'wakeup', 'onset'],
            [0, 1, 2],
            [1, 2, 1]
        ),
        pytest.param(
            [75, 375, 1000],
            ['onset', 'wakeup', 'onset'],
            [1, 2, 0],
            [1, 2, 1]
        ),
        pytest.param(
            [75, 250, 300, 400, 1000],
            ['onset', 'wakeup', 'onset', 'wakeup', 'onset'],
            [1, 0, 0, 2, 0],
            [1, 2, 1, 2, 1]
        )
    ]
)
def test_calc_accuracy_fn(step, event, expected_preds, expected_events):
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
    p, r, f1, preds, events = \
        calc_accuracy(
            preds=preds,
            events=events)
    assert preds == expected_preds
    assert events == expected_events


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

    p, r, f1, preds, events = \
        calc_accuracy(
            preds=preds,
            events=events)
    assert preds == [1, 0, 0, 0]
    assert events == [0, 1, 2, 1]


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
    p, r, f1, preds, events = \
        calc_accuracy(
            preds=preds,
            events=events)
    assert preds == [1, 0]
    assert events == [1, 2]
#
#
# def test_calc_accuracy_multiple_event_matches():
#     preds = pd.DataFrame({
#         'series_id': ['series1', 'series1', 'series1'],
#         'pred': ['onset', 'wakeup', 'onset'],
#         'start': [25, 55, 65],
#         'end': [70, 125, 200]
#     })
#
#     events = pd.DataFrame({
#         'series_id': ['series1', 'series1', 'series1'],
#         'night': [1, 1, 1],
#         'event': ['onset', 'wakeup', 'onset'],
#         'step': [50, 75, 100]
#     })
#     with (tempfile.TemporaryDirectory() as tmpdir):
#         events.to_csv(Path(tmpdir) / 'events.csv', index=False)
#         p, r, f1, preds, events = \
#             calc_accuracy(
#                 preds=preds,
#                 events_path=str(Path(tmpdir) / 'events.csv'))
#         assert preds == [1, 2, 1]
#         assert events == [1, 2, 0]
