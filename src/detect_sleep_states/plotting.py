from typing import List, Dict

import pandas as pd
import torch
from matplotlib import pyplot as plt

from detect_sleep_states.dataset import Label


def plot_predictions(
    data: torch.tensor,
    pred: torch.tensor,
    target: torch.tensor,
    events: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    variable='anglez',
    figsize=(20, 10)
):
    fig, ax = plt.subplots(figsize=figsize)
    colors = {1: 'blue', 2: 'yellow'}

    events = events[(events['start'] >= start_idx) & (events['start'] <= end_idx)]

    anglez = data[0][0]
    enmo = data[0][1]

    label_idx_map = {
        'onset': 1,
        'wakeup': 2
    }

    ax.plot(range(start_idx, end_idx), anglez if variable == 'anglez' else enmo)

    x_axis = torch.ones(pred.shape[0], dtype=torch.float) * -99
    x_axis[torch.where(pred == 1)[0]] = \
    torch.where(pred == 1)[0].type(torch.float)
    x_axis[x_axis == -99] = torch.nan
    x_axis += start_idx

    y_axis = torch.ones(pred.shape[0], dtype=torch.float) * -99
    y_axis[torch.where(pred == 1)[0]] = (
        anglez)[pred == 1] if variable == 'anglez' \
        else enmo[pred == 1]
    y_axis[y_axis == -99] = torch.nan
    ax.plot(x_axis, y_axis, color='purple', label='sleep')

    for label in ('onset', 'wakeup'):
        for event in events[events['event'] == label].itertuples():
            ax.axvline(event.start, color=colors[label_idx_map[label]],
                            linestyle='dashed')
        ax.legend()
    plt.show()
