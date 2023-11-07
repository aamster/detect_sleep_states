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


def dice_loss(
    pred: torch.tensor,
    target: torch.tensor,
    epsilon=1e-6,
    ignore_index=0
):
    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(pred.shape) - 1))

    numerator = 2 * (pred * target).sum(dim=axes)
    denominator = (pred + target).sum(dim=axes)

    # average over classes and batch
    loss = 1 - ((numerator[:, ignore_index+1:] + epsilon) /
                (denominator[:, ignore_index+1:] + epsilon)).mean()

    return loss


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

        self.train_dice_loss = torchmetrics.aggregation.MeanMetric()
        self.val_dice_loss = torchmetrics.aggregation.MeanMetric()

        self._learning_rate = learning_rate
        self._hyperparams = hyperparams

    def training_step(self, batch, batch_idx):
        data, target = batch
        logits = self.model(x=data['sequence'], timestamp_hour=data['hour'])
        probs = torch.softmax(logits, dim=1)
        dl = dice_loss(
            pred=probs.moveaxis(1, 2),  # class axis last
            target=target
        )

        # flattened_target = torch.zeros(target.shape[:-1], dtype=torch.long)
        # for c in range(target.shape[-1]):
        #     flattened_target[torch.where(target[:, :, c] == 1)] = c
        # ce_loss = CrossEntropyLoss()(logits, flattened_target)

        #loss = 0.9 * dl + 0.1 * ce_loss
        loss = dl
        self.train_dice_loss.update(loss)

        self.log("train_dice_loss", dl, prog_bar=True,
                 batch_size=self._batch_size)
        self.logger.log_metrics({
            'train_dice_loss': dl,
        }, step=self.global_step)

        # self.log("train_ce_loss", ce_loss, prog_bar=True,
        #          batch_size=self._batch_size)
        # self.logger.log_metrics({
        #     'train_ce_loss': ce_loss,
        # }, step=self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        logits = self.model(data['sequence'], timestamp_hour=data['hour'])
        probs = torch.softmax(logits, dim=1)
        loss = dice_loss(
            pred=probs.moveaxis(1, 2),  # class axis last
            target=target
        )
        self.val_dice_loss.update(loss)

        self.log('val_dice_loss', loss, on_epoch=True, prog_bar=True,
                 batch_size=self._batch_size)

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
        self.log('train_dice_loss', self.train_dice_loss.compute().item(),
                 on_epoch=True, batch_size=self._batch_size)
        self.train_dice_loss.reset()

    def on_validation_epoch_end(self) -> None:
        self.log('val_dice_loss', self.val_dice_loss.compute().item(), on_epoch=True,
                 batch_size=self._batch_size)
        self.val_dice_loss.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)
        return optimizer
