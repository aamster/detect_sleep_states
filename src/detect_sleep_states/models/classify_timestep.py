from typing import Dict, Any, Literal

import lightning
import torch
import torchmetrics
from torch.nn import CrossEntropyLoss

from detect_sleep_states.dataset import Label


class ClassifyTimestepModel(lightning.pytorch.LightningModule):
    def __init__(
        self,
        learning_rate,
        model: torch.nn.Module,
        hyperparams: Dict,
        batch_size: int,
        return_raw_preds: bool = False,
        small_gap_threshold: int = 720,
        fill_small_gaps: bool = True,
        exclude_invalid_predictions: bool = False,
        step_prediction_method: Literal['max_score', 'middle'] = 'middle'
    ):
        super().__init__()
        self.model = model
        self._batch_size = batch_size
        self._return_raw_preds = return_raw_preds
        self._small_gap_threshold = small_gap_threshold
        self._fill_small_gaps = fill_small_gaps
        self._exclude_invalid_predictions = exclude_invalid_predictions
        self._step_prediction_method = step_prediction_method

        self.train_ce_loss = CrossEntropyLoss(
            weight=torch.tensor([0.9, 0.54, 0.55, 7.8, 7.8])
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
        logits = self.model(data['sequence'], timestamp_hour=data['hour'])
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
            for label in (Label.onset, Label.wakeup):
                # Truncating, in case sequence extends past the actual data
                # and needed to be padded
                preds_possibly_truncated = \
                    preds[batch_idx][:data['sequence_length'][batch_idx]]

                preds_gaps_filled = preds_possibly_truncated

                if self._fill_small_gaps:
                    self.fill_gaps_inference(preds=preds_gaps_filled)

                idxs = torch.where(preds_gaps_filled == label.value)[0]

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

                    # Limiting the score idxs to only those that had the label
                    # before dilation
                    score_idxs = idxs[start:end][
                        torch.where(preds_possibly_truncated[idxs[start:end]] == label.value)]

                    if self._step_prediction_method == 'max_score':
                        # The step is the most confident step in the block
                        step_local = score_idxs[scores[batch_idx][label.value, score_idxs].argmax()]
                    elif self._step_prediction_method == 'middle':
                        # taking average over a sequence of the same block
                        # e.g. multiple contiguous onset predictions, take the
                        # average idx over the block length
                        step_local = idxs[start] + int(
                            (idxs[end - 1] - idxs[start]) / 2)
                    else:
                        raise ValueError(f'Unkown step_prediction_method '
                                         f'{self._step_prediction_method}')

                    # convert to global sequence idx
                    step = data['start'][batch_idx] + step_local

                    if self._exclude_invalid_predictions:
                        # We expect the pattern onset sleep wakeup awake onset
                        # if it is something else, exclude the prediction
                        if idxs[start] > 0:
                            expected_prev_label = (
                                Label.awake.value if label == Label.onset
                                else Label.sleep.value)

                            if preds[batch_idx][idxs[start]-1] != expected_prev_label:
                                continue
                        if idxs[end-1] < preds.shape[1] - 1:
                            expected_next_label = (
                                Label.sleep.value if label == Label.onset
                                else Label.awake.value)
                            if preds[batch_idx][idxs[end-1]+1] != expected_next_label:
                                continue

                    pred_output.append({
                        'series_id': data['series_id'][batch_idx],
                        'event': label.name,
                        'step': step.item(),
                        'score': scores[batch_idx][label.value, score_idxs].mean().item(),
                        'local_start': idxs[start].item(),
                        'local_end': idxs[end-1].item(),
                        'start': (data['start'][batch_idx] + idxs[start]).item(),
                        'end': (data['start'][batch_idx] + idxs[end - 1]).item()
                    })
            preds_output.append(pred_output)

        return preds_output

    def fill_gaps_inference(self, preds: torch.tensor):
        for label in (Label.missing, Label.sleep, Label.awake):
            preds = self._fill_gaps_inference(
                preds=preds,
                label=label
            )
        return preds

    def _fill_gaps_inference(self, preds: torch.tensor, label: Label):
        """
        Fills small gaps in the predicted output
        :return:
        """
        idxs = torch.where(preds == label.value)[0]

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

            if end - start < self._small_gap_threshold:
                if start > 0 and end < preds.shape[0] - 1:
                    if preds[start-1] == preds[end+1]:
                        preds[start:end+1] = preds[start-1]
                    elif preds[start - 1] == Label.missing.value:
                        preds[start:end + 1] = Label.missing.value
                    elif preds[end + 1] == Label.missing.value:
                        preds[start:end + 1] = Label.missing.value
                elif start > 0:
                    if preds[start - 1] == Label.missing.value:
                        preds[start:end + 1] = Label.missing.value
                elif end < preds.shape[0] - 1:
                    if preds[end + 1] == Label.missing.value:
                        preds[start:end + 1] = Label.missing.value
        return preds

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(params=self._hyperparams)

    def on_train_epoch_end(self) -> None:
        f1 = self.train_f1.compute()
        f1 = f1[3:]

        self.log('train_f1', f1.mean(),
                 batch_size=self._batch_size)
        self.log('train_onset_f1', f1[0],
                 batch_size=self._batch_size)
        self.log('train_wakeup_f1', f1[1],
                 batch_size=self._batch_size)
        self.train_f1.reset()

    def on_validation_epoch_end(self) -> None:
        f1 = self.val_f1.compute()
        f1 = f1[3:]

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
