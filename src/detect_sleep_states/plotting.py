from typing import List, Dict

import pandas as pd
import torch
from matplotlib import pyplot as plt

from detect_sleep_states.dataset import Label


def plot_predictions(
    data: torch.tensor,
    pred: torch.tensor,
    target: torch.tensor,
    start_idx: int,
    end_idx: int
):
    fig, ax = plt.subplots(figsize=(20, 10), nrows=2)
    colors = {0: 'blue', 1: 'yellow', 2: 'red', 3: 'green', 4: 'orange'}

    axes = [(pred, 0), (target, 1)]

    anglez = data[0][0]

    for axis_data, idx in axes:
        for label in Label:
            x_axis = torch.ones(axis_data.shape[0], dtype=torch.float) * -99
            x_axis[torch.where(axis_data == label.value)[0]] = torch.where(axis_data == label.value)[0].type(torch.float)
            x_axis[x_axis == -99] = torch.nan
            x_axis += start_idx

            y_axis = torch.ones(axis_data.shape[0], dtype=torch.float) * -99
            y_axis[torch.where(axis_data == label.value)[0]] = anglez[axis_data == label.value]
            y_axis[y_axis == -99] = torch.nan
            ax[idx].plot(x_axis, y_axis, color=colors[label.value], label=label.name)
            ax[idx].legend()
        ax[idx].set_xticks(range(start_idx, end_idx, 1440))
    plt.show()
