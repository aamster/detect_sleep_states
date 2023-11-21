import pytest
import torch.nn
from torch import nn

from detect_sleep_states.models.classify_timestep import ClassifyTimestepModel


class BasicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3,
                               stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class TestClassifyTimestep:
    @classmethod
    def setup_class(cls):
        cls._model = ClassifyTimestepModel(
            learning_rate=1e-3,
            model=BasicCNN(),
            hyperparams={},
            batch_size=8,
            return_raw_preds=False,
            small_gap_threshold=1200,
            exclude_invalid_predictions=True,
            step_prediction_method='max_score'
        )

    @pytest.mark.parametrize(
        'preds, expected', [
            pytest.param(
                torch.tensor([0, 1, 1, 3]),
                torch.tensor([0, 1, 1, 3])
            ),
            pytest.param(
                torch.tensor([3, 3, 1, 3]),
                torch.tensor([3, 3, 3, 3])
            ),
            pytest.param(
                torch.tensor([4, 4, 1, 4]),
                torch.tensor([4, 4, 4, 4])
            ),
            pytest.param(
                torch.tensor([2, 1, 1, 1]),
                torch.tensor([2, 2, 2, 2])
            ),
            pytest.param(
                torch.tensor([1, 2, 2, 2]),
                torch.tensor([2, 2, 2, 2])
            ),
            pytest.param(
                torch.tensor([3, 2, 2, 3]),
                torch.tensor([3, 3, 3, 3])
            ),
            pytest.param(
                torch.tensor([1, 1, 1, 1]),
                torch.tensor([1, 1, 1, 1])
            ),
            pytest.param(
                torch.tensor([3, 3, 3, 3]),
                torch.tensor([3, 3, 3, 3])
            ),
            pytest.param(
                torch.tensor([3, 3, 1, 3, 2, 2, 3, 3]),
                torch.tensor([3, 3, 3, 3, 3, 3, 3, 3])
            ),
            pytest.param(
                torch.tensor([0, 3, 1]),
                torch.tensor([0, 3, 1])
            ),
            pytest.param(
                torch.tensor([0, 3, 2, 1]),
                torch.tensor([0, 3, 2, 2])
            ),
            pytest.param(
                torch.tensor([3, 0, 0, 2]),
                torch.tensor([3, 2, 2, 2])
            ),
        ]
    )
    def test__fill_gaps_inference(self, preds, expected):
        preds_filled = self._model._fill_gaps_inference(preds=preds)
        assert (preds_filled == expected).all().item()
