from typing import Dict, Any

import lightning
import torch
import torchmetrics

from detect_sleep_states.classify_segment.dataset import label_id_str_map


class ClassifySegmentModel(lightning.LightningModule):
    def __init__(
        self,
        learning_rate,
        model: torch.nn.Module,
        hyperparams: Dict,
        batch_size: int
    ):
        super().__init__()
        num_classes = len(label_id_str_map)
        self._loss_function = torch.nn.CrossEntropyLoss()
        self.model = model
        self._batch_size = batch_size

        self.val_precision = torchmetrics.classification.MulticlassPrecision(
            num_classes=num_classes)
        self.val_recall = torchmetrics.classification.MulticlassRecall(
            num_classes=num_classes
        )
        self.train_f1 = torchmetrics.classification.MulticlassF1Score(
            num_classes=num_classes
        )
        self.val_f1 = torchmetrics.classification.MulticlassF1Score(
            num_classes=num_classes
        )
        self._learning_rate = learning_rate
        self._hyperparams = hyperparams

    def training_step(self, batch, batch_idx):
        data, target = batch
        logits = self.model(x=data['sequence'], start_hour=data['start_hour'])
        loss = self._loss_function(logits, target)
        preds = torch.argmax(logits, dim=1)

        self.train_f1.update(preds, target)
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
        self.val_precision.update(preds=preds, target=target)
        self.val_recall.update(preds=preds, target=target)
        self.val_f1.update(preds=preds, target=target)

    def predict_step(
            self,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0
    ):
        data, target = batch
        logits = self.model(data['sequence'])
        preds = torch.argmax(logits, dim=1)

        return [label_id_str_map[pred.item()] for pred in preds]

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(params=self._hyperparams)

    def on_train_epoch_end(self) -> None:
        self.log('train_f1', self.train_f1.compute().item(),
                 on_epoch=True, batch_size=self._batch_size)
        self.train_f1.reset()

    def on_validation_epoch_end(self) -> None:
        self.log('val_f1', self.val_f1.compute().item(), on_epoch=True,
                 batch_size=self._batch_size)
        self.log('val_precision', self.val_precision.compute().item(),
                 on_epoch=True, batch_size=self._batch_size)
        self.log('val_recall', self.val_recall.compute().item(),
                 on_epoch=True, batch_size=self._batch_size)
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)
        return optimizer
