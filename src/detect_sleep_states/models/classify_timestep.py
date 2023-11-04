from typing import Dict, Any

import lightning
import torch
import torchmetrics
from torch import nn
from torchvision.ops.misc import ConvNormActivation

from detect_sleep_states.dataset import label_id_str_map


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
        self._loss_function = torch.nn.CrossEntropyLoss()
        self.model = model
        self._batch_size = batch_size

        self.train_aucroc = torchmetrics.classification.BinaryAUROC()
        self.val_aucroc = torchmetrics.classification.BinaryAUROC()

        self._learning_rate = learning_rate
        self._hyperparams = hyperparams

    def training_step(self, batch, batch_idx):
        data, target = batch
        logits = self.model(x=data['sequence'], start_hour=data['start_hour'])
        loss = self._loss_function(logits, target)
        preds = torch.argmax(logits, dim=1)

        self.train_aucroc.update(preds, target)
        self.log("train_loss", loss, prog_bar=True,
                 batch_size=self._batch_size)
        self.logger.log_metrics({
            'train_loss': loss,
        }, step=self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        logits = self.model(data['sequence'], start_hour=data['start_hour'])
        loss = self._loss_function(logits, target)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True,
                 batch_size=self._batch_size)

        preds = torch.argmax(logits, dim=1)
        self.val_aucroc.update(preds=preds, target=target)

    def predict_step(
            self,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0
    ):
        data = batch
        logits = self.model(data['sequence'], start_hour=data['start_hour'])
        preds = torch.argmax(logits, dim=1)
        scores = torch.nn.functional.softmax(logits, dim=1)
        res = [{
            'pred': label_id_str_map[preds[i].item()],
            'scores': scores[i],
            'start': data['start'][i].item(),
            'end': data['end'][i].item()
        } for i in range(len(preds))]
        return res

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(params=self._hyperparams)

    def on_train_epoch_end(self) -> None:
        self.log('train_aucroc', self.train_aucroc.compute().item(),
                 on_epoch=True, batch_size=self._batch_size)
        self.train_aucroc.reset()

    def on_validation_epoch_end(self) -> None:
        self.log('val_aucroc', self.val_aucroc.compute().item(), on_epoch=True,
                 batch_size=self._batch_size)
        self.val_aucroc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)
        return optimizer
