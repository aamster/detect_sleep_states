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
    fig, ax = plt.subplots(figsize=figsize, nrows=2)
    colors = {0: 'blue', 1: 'yellow', 2: 'red', 3: 'green', 4: 'orange'}

    events = events[(events['start'] >= start_idx) & (events['start'] <= end_idx)]

    axes = [(pred, 0), (target, 1)]

    anglez = data[0][0]
    enmo = data[0][1]

    for axis_data, idx in axes:
        for label in Label:
            x_axis = torch.ones(axis_data.shape[0], dtype=torch.float) * -99
            x_axis[torch.where(axis_data == label.value)[0]] = torch.where(axis_data == label.value)[0].type(torch.float)
            x_axis[x_axis == -99] = torch.nan
            x_axis += start_idx

            y_axis = torch.ones(axis_data.shape[0], dtype=torch.float) * -99
            y_axis[torch.where(axis_data == label.value)[0]] = (
                anglez)[axis_data == label.value] if variable == 'anglez' \
                else enmo[axis_data == label.value]
            y_axis[y_axis == -99] = torch.nan
            ax[idx].plot(x_axis, y_axis, color=colors[label.value], label=label.name)

            if label in (Label.onset, Label.wakeup):
                for event in events[events['event'] == label.name].itertuples():
                    ax[idx].axvline(event.start, color=colors[label.value],
                                    linestyle='dashed')
            ax[idx].legend()
    plt.show()
