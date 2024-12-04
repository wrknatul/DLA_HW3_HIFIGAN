import torch 
from itertools import chain

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer

from src.datasets.mel_generator import MelSpectrogramConfig, MelSpectrogram


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer_disk.zero_grad()
            self.optimizer_gen.zero_grad()

        batch["wavs_predictions"] = self.model.generator(batch["mels"])
        mel_transform = MelSpectrogram(MelSpectrogramConfig)
        batch["mels_predictions"] = mel_transform(batch["wavs_predictions"]).squeeze(1)

        mpd_real, _ = self.model.mpd(batch["wavs"])
        mpd_predicted, _ = self.model.mpd(batch["wavs_predictions"].detach())
        msd_real, _ = self.model.msd(batch["wavs"])
        msd_prediction, _ = self.model.msd(batch["wavs_predictions"].detach())

        batch["disk_loss"] = self.criterion.discriminatorLoss(mpd_real, mpd_predicted) + self.criterion.discriminatorLoss(msd_real, msd_prediction)

        if self.is_train:
            batch["disk_loss"].backward()
            self._clip_grad_norm(
                chain(self.model.mpd.parameters(), self.model.msd.parameters())
            )
            self.optimizer_disk.step()
            self.lr_scheduler_disk.step()

        mpd_wavs_real, mpd_wavs_feat = self.model.mpd(batch["wavs"])
        mpd_pred_real, mpd_pred_feat = self.model.mpd(batch["wavs_predictions"].detach())
        msd_wavs_real, msd_wavs_feat = self.model.msd(batch["wavs"])
        msd_pred_real, msd_pred_feat = self.model.msd(batch["wavs_predictions"].detach())

        mpd_gen_loss = self.criterion.generatorLoss(mpd_pred_real)
        msd_gen_loss = self.criterion.generatorLoss(msd_pred_real)
        mpd_feat_loss = self.criterion.featureMatchingLoss(mpd_wavs_feat, mpd_pred_feat)
        msd_feat_loss = self.criterion.featureMatchingLoss(msd_wavs_feat, msd_pred_feat)
        mel_loss = self.criterion.melSpectrogramLoss(batch["mels"], batch["mels_predictions"])

        batch["gen_loss"] = mpd_gen_loss + msd_gen_loss
        batch["feature_loss"] = 2 * (mpd_feat_loss + msd_feat_loss)
        batch["mel_loss"] = 45 * mel_loss
        batch["total_loss"] = batch["gen_loss"] + batch["feature_loss"] + batch["mel_loss"]

        if self.is_train:
            batch["total_loss"].backward()
            self._clip_grad_norm(self.model.generator.parameters())
            self.optimizer_gen.step()
            self.lr_scheduler_gen.step()

        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met, batch[met].item())
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # Log Stuff
            pass
        else:
            # Log Stuff
            pass
