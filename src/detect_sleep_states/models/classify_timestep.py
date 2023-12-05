from typing import Dict, Any, Optional, List

import lightning
import pandas as pd
import torch
import torchmetrics
from monai.losses import DiceLoss
from torch import nn
from torch.nn import BCEWithLogitsLoss
from unet import UNet1D
from unet.conv import ConvolutionalBlock

from detect_sleep_states.dataset import Label
from detect_sleep_states.metric import score


class CNN(UNet1D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        with torch.no_grad():
            skip_connections, encoding = self.encoder(x)
            encoding = self.bottom_block(encoding)
        return encoding, skip_connections


class CNNRNN(nn.Module):
    def __init__(
        self,
        rnn_hidden_size: int,
        cnn_weights_path: str,
        rnn_bidirectional: bool = False,
        num_rnn_layers: int = 1,
        freeze_cnn: bool = False
    ):
        super().__init__()
        self.unet1d = CNN(
            in_channels=2,
            out_classes=len(Label),
            padding=1,
            normalization='batch',
            residual=True,
            num_encoding_blocks=5,
            out_channels_first_layer=16,
            kernel_size=51
        )
        if freeze_cnn:
            for param in self.unet1d.encoder.parameters():
                param.requires_grad = False
            for param in self.unet1d.bottom_block.parameters():
                param.requires_grad = False

        map_location = torch.device('cpu') if not torch.cuda.is_available() else None
        cnn = ClassifyTimestepModel.load_from_checkpoint(
            checkpoint_path=cnn_weights_path,
            map_location=map_location,
            learning_rate=1e-3,
            model=self.unet1d,
            hyperparams={},
            batch_size=8
        ).model

        self.cnn = cnn
        self.rnn = nn.LSTM(
            input_size=256,  # num channels output by CNN
            hidden_size=rnn_hidden_size,
            batch_first=True,
            bidirectional=rnn_bidirectional,
            num_layers=num_rnn_layers
        )
        self.fc = nn.Linear(self.rnn.hidden_size*num_rnn_layers, 256)
        self.classifier = ConvolutionalBlock(
            dimensions=1,
            in_channels=16 + 24,
            out_channels=len(Label),
            kernel_size=1,
            activation=None
        )

    def forward(self, x, timestamp_hour: torch.tensor):
        x, skip_connections = self.cnn(x)

        # Preparing for RNN (N: batch size, T: sequence length, C: number of channels)
        x = x.permute(0, 2, 1)

        x, _ = self.rnn(x)

        x = self.fc(x)

        x = self.unet1d.decoder(skip_connections, x.moveaxis(2, 1))

        timestamp_hour = nn.functional.one_hot(timestamp_hour.long(),
                                               num_classes=24)
        timestamp_hour = timestamp_hour.moveaxis(2, 1)

        x = torch.cat([x, timestamp_hour], 1)

        x = self.classifier(x)

        return x


class ClassifyTimestepModel(lightning.pytorch.LightningModule):
    def __init__(
        self,
        learning_rate,
        model: torch.nn.Module,
        hyperparams: Dict,
        batch_size: int,
        return_raw_preds: bool = False,
        small_gap_threshold: int = 360,
        fill_small_gaps: bool = True,
        exclude_invalid_predictions: bool = False,
        events_path: Optional[str] = None
    ):
        super().__init__()
        self.model = model
        self._batch_size = batch_size
        self._return_raw_preds = return_raw_preds
        self._small_gap_threshold = small_gap_threshold
        self._fill_small_gaps = fill_small_gaps
        self._exclude_invalid_predictions = exclude_invalid_predictions
        self._events_path = events_path

        self._loss_fn = BCEWithLogitsLoss()

        self._learning_rate = learning_rate
        self._hyperparams = hyperparams

    def training_step(self, batch, batch_idx):
        data, target = batch
        logits = self.model(x=data['sequence'].float())

        loss = self._loss_fn(logits, target.moveaxis(2, 1))

        self.log("train_bce_loss", loss, prog_bar=True,
                 batch_size=self._batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        logits = self.model(data['sequence'].float())

        loss = self._loss_fn(logits, target.moveaxis(2, 1))

        self.log(
            "val_bce_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self._batch_size
        )

    @staticmethod
    def _flatten_preds_and_targets(probs, target, threshold=0.5):
        flattened_target = ClassifyTimestepModel.flatten_targets(target=target)

        flattened_preds = torch.zeros((probs.shape[0], probs.shape[-1]),
                                      dtype=torch.long,
                                      device=probs.device
                                      )
        for c in range(probs.shape[1]):
            flattened_preds[torch.where(probs[:, c] > threshold)] = c
        return flattened_preds, flattened_target

    @staticmethod
    def flatten_targets(target):
        flattened_target = torch.zeros(target.shape[:-1], dtype=torch.long,
                                       device=target.device)
        for c in range(target.shape[-1]):
            flattened_target[torch.where(target[:, :, c] == 1)] = c
        return flattened_target

    def predict_step(
            self,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0
    ):
        if len(batch) == 2:
            data, _ = batch
        else:
            data = batch
        logits = self.model(data['sequence'].float())
        scores = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        if self._return_raw_preds:
            return data['sequence'], preds
        # Taking the raw preds which can contain multiple contiguous e.g.
        # wakeup predictions and finding the middle, then converting to
        # a global index
        preds_output = []
        for batch_idx in range(preds.shape[0]):
            pred_output = []
            # Truncating, in case sequence extends past the actual data
            # and needed to be padded
            preds_possibly_truncated = \
                preds[batch_idx][:data['sequence_length'][batch_idx]]

            if self._fill_small_gaps:
                preds_gaps_filled = self.fill_gaps_inference(
                    preds=preds_possibly_truncated)
            else:
                preds_gaps_filled = preds_possibly_truncated

            idxs = torch.where(preds_gaps_filled == 1)[0]

            if len(idxs) == 0:
                continue

            i = 0
            while i < len(idxs):
                start = i
                prev_idx = idxs[i]
                while (i < len(idxs) and
                       (idxs[i] == prev_idx or idxs[i] == prev_idx + 1)):
                    prev_idx = idxs[i]
                    i += 1
                end = i

                score_idxs = idxs[start:end]

                steps_local = [idxs[start], idxs[end - 1]]

                # convert to global sequence idx
                steps = [data['start'][batch_idx] + x for x in steps_local]

                if self._exclude_invalid_predictions:
                    # We expect the pattern onset sleep wakeup awake onset
                    # if it is something else, exclude the prediction
                    raise NotImplementedError()
                    # if idxs[start] > 0:
                    #     expected_prev_label = (
                    #         Label.awake.value if label == Label.onset
                    #         else Label.sleep.value)
                    #
                    #     if preds[batch_idx][idxs[start]-1] != expected_prev_label:
                    #         continue
                    # if idxs[end-1] < preds.shape[1] - 1:
                    #     expected_next_label = (
                    #         Label.sleep.value if label == Label.onset
                    #         else Label.awake.value)
                    #     if preds[batch_idx][idxs[end-1]+1] != expected_next_label:
                    #         continue

                if steps_local[-1] - steps_local[0] > 360:
                    for step, step_label in zip(steps, ('onset', 'wakeup')):
                        score_idx = 0 if step_label == 'onset' else -1
                        pred_output.append({
                            'series_id': data['series_id'][batch_idx],
                            'event': step_label,
                            'step': step.item(),
                            'score': scores[batch_idx][1, score_idxs[score_idx]].item()
                        })
            preds_output.append(pred_output)

        return preds_output

    def fill_gaps_inference(self, preds: torch.tensor):
        """
        Fills small gaps in the predicted output
        :return:
        """
        segments = []
        idxs = torch.where(preds == 1)[0]

        i = 0
        while i < len(idxs):
            start = i
            prev_idx = idxs[i]
            while (i < len(idxs) and
                   (idxs[i] == prev_idx or idxs[i] == prev_idx + 1)):
                prev_idx = idxs[i]
                i += 1
            end = i

            start = idxs[start]
            end = idxs[end-1]

            if len(segments) > 0:
                prev_end = segments[-1][1]
                if end - self._small_gap_threshold < prev_end:
                    prev_start = segments[-1][0]
                    segments[-1] = (prev_start, end)
                else:
                    segments.append((start, end))
            else:
                segments.append((start, end))

        for segment_start, segment_end in segments:
            preds[segment_start:segment_end+1] = 1
        return preds

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(params=self._hyperparams)

    def on_validation_epoch_end(self) -> None:
        pass

    def _calculate_map(self, preds: List, series_ids: set) -> float:
        submission_flat = []
        for batch in preds:
            for sequence_pred in batch:
                for pred in sequence_pred:
                    submission_flat.append(pred)
        preds = pd.DataFrame(submission_flat)

        solution = pd.read_csv(self._events_path)
        solution = solution[
            solution['series_id'].isin(list(series_ids))]
        solution = solution[~solution['step'].isna()]

        if preds.empty:
            preds = pd.DataFrame(
                columns=list(solution.columns) + ['score']
            )
            preds['step'] = preds['step'].astype(int)
            preds['score'] = preds['score'].astype(float)
            return 0.0

        column_names = {
            'series_id_column_name': 'series_id',
            'time_column_name': 'step',
            'event_column_name': 'event',
            'score_column_name': 'score',
        }
        tolerances = {'onset': [12, 36, 60, 90, 120, 150, 180, 240, 300, 360],
                      'wakeup': [12, 36, 60, 90, 120, 150, 180, 240, 300, 360]}
        map, detections_matched, gt_matched = score(
            solution, preds, tolerances, **column_names)

        return map

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)
        return optimizer
