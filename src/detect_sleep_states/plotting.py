from typing import List, Dict

import torch
from matplotlib import pyplot as plt

from detect_sleep_states.dataset import Label


def plot_predictions(
    data: torch.tensor,
    pred: torch.tensor
):
    fig, ax = plt.subplots(figsize=(20, 10))
    colors = {0: 'blue', 1: 'yellow', 2: 'red', 3: 'green', 4: 'orange'}
    for label in Label:
        x_axis = torch.ones(pred.shape[0], dtype=torch.float) * -99
        x_axis[torch.where(pred == label.value)[0]] = \
        torch.where(pred == label.value)[0].type(torch.float)
        x_axis[x_axis == -99] = torch.nan

        y_axis = torch.ones(pred.shape[0], dtype=torch.float) * -99
        y_axis[torch.where(pred == label.value)[0]] = \
        data[0][pred == label.value]
        y_axis[y_axis == -99] = torch.nan
        ax.plot(x_axis, y_axis, color=colors[label.value], label=label.name)
    plt.legend()
    plt.show()
