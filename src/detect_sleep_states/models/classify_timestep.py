from typing import Dict, Any

import lightning
import torch
import torchmetrics
from torch import nn
from torch.nn import CrossEntropyLoss
from torchvision.ops.misc import ConvNormActivation

from detect_sleep_states.dataset import label_id_str_map, Label


class SamePaddingTimestepPredictor1dCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            ConvNormActivation(
                in_channels=2,
                out_channels=64,
                kernel_size=49,
                bias=False,
                norm_layer=nn.BatchNorm1d,
                conv_layer=nn.Conv1d,
                stride=1,
                padding='same'
            ),
            ConvNormActivation(
                in_channels=64,
                out_channels=128,
                kernel_size=49,
                bias=False,
                norm_layer=nn.BatchNorm1d,
                conv_layer=nn.Conv1d,
                stride=1,
                padding='same'
            ),
            ConvNormActivation(
                in_channels=128,
                out_channels=256,
                kernel_size=49,
                bias=False,
                norm_layer=nn.BatchNorm1d,
                conv_layer=nn.Conv1d,
                stride=1,
                padding='same'
            )
        )

        self.classifier = nn.Conv1d(
            in_channels=256+24,
            out_channels=len(label_id_str_map),
            kernel_size=1
        )

    def forward(self, x, start_hour):
        x = self.encoder(x)

        start_hour = nn.functional.one_hot(start_hour.long(), num_classes=24)
        start_hour = start_hour.unsqueeze(2).repeat(1, 1, x.shape[-1])

        x = torch.cat([x, start_hour], 1)
        x = self.classifier(x)

        return x


class ClassifyTimestepModel(lightning.pytorch.LightningModule):
    def __init__(
        self,
        learning_rate,
        model: torch.nn.Module,
        hyperparams: Dict,
        batch_size: int
    ):
        super().__init__()
        self.model = model
        self._batch_size = batch_size

        self.train_ce_loss = CrossEntropyLoss(
            weight=torch.tensor([0.35, 13.0, 13.0])
        )
        self.train_f1 = torchmetrics.classification.MulticlassF1Score(
            num_classes=len(Label),
            average='none'
        )
        self.val_f1 = torchmetrics.classification.MulticlassF1Score(
            num_classes=len(Label),
            average='none'
        )

        self._learning_rate = learning_rate
        self._hyperparams = hyperparams

    def training_step(self, batch, batch_idx):
        data, target = batch
        logits = self.model(x=data['sequence'], timestamp_hour=data['hour'])
        probs = torch.softmax(logits, dim=1)

        flattened_preds, flattened_target = self._flatten_preds_and_targets(
            probs=probs,
            target=target
        )

        loss = self.train_ce_loss(logits, flattened_target)

        self.log("train_ce_loss", loss, prog_bar=True,
                 batch_size=self._batch_size)
        self.train_f1.update(
            preds=flattened_preds,
            target=flattened_target
        )
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        logits = self.model(data['sequence'], timestamp_hour=data['hour'])
        probs = torch.softmax(logits, dim=1)

        flattened_preds, flattened_target = self._flatten_preds_and_targets(
            probs=probs,
            target=target
        )

        self.val_f1.update(
            preds=flattened_preds,
            target=flattened_target
        )

    @staticmethod
    def _flatten_preds_and_targets(probs, target, threshold=0.5):
        flattened_target = torch.zeros(target.shape[:-1], dtype=torch.long,
                                       device=target.device)
        for c in range(target.shape[-1]):
            flattened_target[torch.where(target[:, :, c] == 1)] = c

        flattened_preds = torch.zeros((probs.shape[0], probs.shape[-1]),
                                      dtype=torch.long,
                                      device=probs.device
                                      )
        for c in range(probs.shape[1]):
            flattened_preds[torch.where(probs[:, c] > threshold)] = c
        return flattened_preds, flattened_target

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
        logits = self.model(data['sequence'], timestamp_hour=data['hour'])
        preds = torch.argmax(logits, dim=1)

        # Taking the raw preds which can contain multiple contiguous e.g.
        # wakeup predictions and finding the middle, then converting to
        # a global index
        preds_output = []
        for pred_idx in range(preds.shape[0]):
            pred_output = []
            i = 0
            while i < preds.shape[1]:
                pred = preds[pred_idx][i]
                start = i
                while i < preds.shape[1] and preds[pred_idx][i] == pred:
                    i += 1
                end = i
                if pred == Label.onset.value:
                    pred = Label.onset.name
                elif pred == Label.wakeup.value:
                    pred = Label.wakeup.name
                else:
                    continue
                # taking average over a sequence of the same block
                # e.g. multiple contiguous onset predictions, take the
                # average idx over the block length
                step_local = start + int((end - start) / 2)

                # convert to global sequence idx
                step = data['start'][pred_idx] + step_local

                pred_output.append({
                    'series_id': data['series_id'][pred_idx],
                    'event': pred,
                    'step': step.item()
                })
            preds_output.append(pred_output)

        return preds_output

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(params=self._hyperparams)

    def on_train_epoch_end(self) -> None:
        f1 = self.train_f1.compute()
        f1 = f1[1:]

        self.log('train_f1', f1.mean(),
                 batch_size=self._batch_size)
        self.log('train_onset_f1', f1[0],
                 batch_size=self._batch_size)
        self.log('train_wakeup_f1', f1[1],
                 batch_size=self._batch_size)
        self.train_f1.reset()

    def on_validation_epoch_end(self) -> None:
        f1 = self.val_f1.compute()
        f1 = f1[1:]

        self.log('val_f1', f1.mean(),
                 batch_size=self._batch_size)
        self.log('val_onset_f1', f1[0],
                 batch_size=self._batch_size)
        self.log('val_wakeup_f1', f1[1],
                 batch_size=self._batch_size)
        self.val_f1.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)
        return optimizer
