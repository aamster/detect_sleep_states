import pandas as pd
import torch

from detect_sleep_states.dataset import ClassifySegmentDataset, Label


def calc_target_weights(train_meta: pd.DataFrame,
                        train_events: pd.DataFrame,
                        sequence_length: int):
    """Returns weights for loss such that each target is equally considered
    during training"""
    total_counts = torch.zeros(len(Label))
    for row in train_meta.itertuples():
        label = ClassifySegmentDataset.construct_target(
            events=train_events,
            series_id=row.Index,
            start=row.start,
            sequence_length=sequence_length)
        counts = label.sum(dim=0)
        total_counts += counts

    N = (train_meta['end'] - train_meta['start']).sum()
    C = len(Label)
    weights = N / (C * total_counts)
    return weights
