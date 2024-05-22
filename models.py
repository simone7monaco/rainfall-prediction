import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from PIL import Image
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

from utils import io

torch.set_float32_matmul_precision("high")
NUM_WORKERS = 64

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from networks.unet import ExtraUNet, UNet
from networks.vgunet import VGUNet
from utils.datasets import NWPDataset


class SegmentationModel(pl.LightningModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()

        self.load_data()
        if self.hparams.network_model == "unet":
            self.cnn = UNet(self.in_features, self.out_features, dropout=0)
        elif self.hparams.network_model == "vgunet":
            self.cnn = VGUNet(self.in_features, self.out_features, base_nc=64)
        elif self.hparams.network_model.startswith("extra"):
            self.cnn = ExtraUNet(
                self.in_features,
                self.out_features,
                image_shape=(self.x_train[0].shape[1], self.x_train[0].shape[2]),
                use_attention=True,
            )
        else:
            raise NotImplementedError(
                f"Model {self.hparams.network_model} not implemented"
            )
        #self.MSE = nn.MSELoss()
        self.sigmoid = nn.Sigmoid()
        self.BCEL = nn.BCEWithLogitsLoss()
        self.BCE = nn.BCELoss()
        self.FCL = FocalLoss()
        #self.ERL = EntropyRegularizedLoss()
        # self.loss = lambda y_hat, y: F.mse_loss(y_hat * self.mask, y * self.mask)

        self.rmse = lambda loss: (loss * (self.case_study_max**2)).sqrt().item()
        thresh = [
            1 / self.case_study_max,
            5 / self.case_study_max,
            10 / self.case_study_max,
            20 / self.case_study_max,
            50 / self.case_study_max,
            100 / self.case_study_max,
            150 / self.case_study_max,
        ]
        self.thtot = [
            1, 5, 10, 20, 50, 100, 150
        ]
        thresholds_indx = [x%self.hparams.n_thresh for x in range(self.hparams.indx_thresh, self.hparams.n_thresh+self.hparams.indx_thresh)]
        self.thresholds = [thresh[indx] for indx in thresholds_indx]
        self.metrics = [ECE, KL, AUC, brierScore]
        self.test_predictions = []

        self.train_losses = []
        self.val_losses = []

    def forward(self, x, times):
        if isinstance(self.cnn, ExtraUNet):
            return self.cnn(x, times)
        y = self.cnn(x) * self.mask
        y_prob = self.sigmoid(y) * self.mask
        return y, y_prob

    def load_data(self):
        (
            case_study_max,
            available_models,
            train_dates,
            val_dates,
            test_dates,
            indices_one,
            indices_zero,
            mask,
            nx,
            ny,
        ) = io.get_casestudy_stuff(
            self.hparams.input_path,
            n_split=self.hparams.n_split,
            case_study=self.hparams.case_study,
            ispadded=True,
            seed=self.hparams.seed,
        )
        self.x_train, self.y_train, in_features, out_features = io.load_data(
            self.hparams.input_path,
            train_dates,
            case_study_max,
            indices_one,
            indices_zero,
            available_models,
        )
        self.x_val, self.y_val, in_features, out_features = io.load_data(
            self.hparams.input_path,
            val_dates,
            case_study_max,
            indices_one,
            indices_zero,
            available_models,
        )
        self.x_test, self.y_test, in_features, out_features = io.load_data(
            self.hparams.input_path,
            test_dates,
            case_study_max,
            indices_one,
            indices_zero,
            available_models,
        )

        self.train_dates, self.val_dates, self.test_dates = (
            train_dates,
            val_dates,
            test_dates,
        )

        self.case_study_max = case_study_max
        mask = torch.from_numpy(mask).float().to(self.device)
        self.register_buffer(
            "mask", mask
        )  # This makes sure self.mask is on the same device as the model

        self.in_features = in_features
        self.out_features = self.hparams.n_thresh  # out_features

    def train_dataloader(self):
        if isinstance(self.cnn, ExtraUNet):
            self.train_dates = io.date_features(self.train_dates)
        train_dataset = NWPDataset(
            (
                torch.from_numpy(self.x_train),
                torch.from_numpy(self.y_train).unsqueeze(1),
                torch.from_numpy(self.train_dates),
            )
        )
        return DataLoader(
            train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
        )

    def val_dataloader(self):
        if isinstance(self.cnn, ExtraUNet):
            self.val_dates = io.date_features(self.val_dates)
        val_dataset = NWPDataset(
            (
                torch.from_numpy(self.x_val),
                torch.from_numpy(self.y_val).unsqueeze(1),
                torch.from_numpy(self.val_dates),
            )
        )
        return DataLoader(
            val_dataset,
            batch_size=len(val_dataset),
            shuffle=False,
            num_workers=NUM_WORKERS,
        )

    def test_dataloader(self):
        if isinstance(self.cnn, ExtraUNet):
            self.test_dates = io.date_features(self.test_dates)
        test_dataset = NWPDataset(
            (
                torch.from_numpy(self.x_test),
                torch.from_numpy(self.y_test).unsqueeze(1),
                torch.from_numpy(self.test_dates),
            )
        )
        return DataLoader(
            test_dataset,
            batch_size=len(test_dataset),
            shuffle=False,
            num_workers=NUM_WORKERS,
        )

    def get_new_prob(self, mean_value, prob_array, true_array, scale):
        weight = scipy.stats.norm.pdf(prob_array, loc=mean_value, scale=scale)
        return np.sum(weight * true_array) / np.sum(weight)

    def training_step(self, batch, batch_idx):
        x, y, ev_date = batch["x"], batch["y"], batch.get("ev_date")
        y_hat, y_hat_prob = self.forward(
            x, ev_date
        )  # shape (n_repetitions*n_samples, C, H, W)
        y_p = []
        for i in range(len(self.thresholds)):
            y_p.append(y.gt(self.thresholds[i]).float())
        y_p = torch.cat(y_p, dim=1)
        loss_CAPE = 0
        loss_BCE = 0
        if self.hparams.fine_tune == 1 and self.current_epoch %2==0:
            n_bins = 20
            if (
                self.hparams.finetune_type == "bin"
                or self.hparams.finetune_type == "kde"
            ):
                for i in range(len(self.thresholds)):
                    # sort to calculate bins
                    y_hat_prob_mask = y_hat_prob[:, i, self.mask == 1].flatten()
                    sorted_idx = torch.argsort(y_hat_prob_mask)  ########
                    targets_probs = y_hat_prob_mask[sorted_idx]
                    labels = y_p[:, i, self.mask == 1].flatten()
                    labels = labels[sorted_idx]
                    num_sample = len(labels)
                    # indices = torch.arange(num_sample).to(self.device)
                    # indices = indices[sorted_idx]
                    # flat_mask = self.mask.flatten()
                    # num_mask = len(flat_mask)*y_p.size(0)*y_p.size(1) #mask*n_sample*n_layer
                    # proposed_probs = torch.zeros(num_mask).to(self.device)
                    new_labels = torch.zeros(num_sample).to(self.device)
                    if self.hparams.finetune_type == "bin":
                        for i in range(n_bins):
                            left = int(i * num_sample / n_bins)
                            right = int((i + 1) * num_sample / n_bins)
                            new_labels[left:right] = torch.mean((labels[left:right]))
                    elif self.hparams.finetune_type == "kde":
                        targets_probs_np = targets_probs.detach().cpu().numpy()
                        labels = labels.detach().cpu().numpy()
                        sigma = 0.1
                        window = 500
                        for i in range(num_sample):
                            left = np.maximum(0, i - window)
                            right = np.minimum(i + window, num_sample)
                            new_labels[i] = self.get_new_prob(
                                targets_probs_np[i],
                                targets_probs_np[left:right],
                                labels[left:right],
                                scale=sigma,
                            )
                        # new_labels = torch.from_numpy(new_labels).to(self.device)
                    # j=0
                    # for i in range(num_mask):
                    # 	if(flat_mask[i%len(self.mask)] == 1):
                    # 		proposed_probs[int(indices[j])] = new_labels[j]
                    # 		j+=1
                    # probs_emp = torch.reshape(proposed_probs, (y_p.size(0), y_p.size(1), y_p.size(2), y_p.size(3)))
                    loss_CAPE = loss_CAPE + self.BCE(targets_probs, new_labels)
                loss_CAPE = loss_CAPE / len(self.thresholds)
            elif self.hparams.finetune_type == "mine":
                probs_emp = torch.zeros(
                    [y_p.size(0), y_p.size(1), y_p.size(2), y_p.size(3)]
                ).to(self.device)
                probs_mask = y_hat_prob
                bins = torch.linspace(1 / n_bins, 1, n_bins).to(self.device)
                bins_index = torch.bucketize(probs_mask, bins, right=True).to(
                    self.device
                )
                for i in range(n_bins):
                    inx = torch.where(bins_index == i)
                    probs_emp[inx] = torch.mean(y_p[inx])
                loss_CAPE = self.BCEL(
                    y_hat[:, :, self.mask == 1], probs_emp[:, :, self.mask == 1]
                )
            else:
                raise NotImplementedError

        else:
            loss_BCE = self.BCEL(y_hat[:, :, self.mask == 1], y_p[:, :, self.mask == 1])
        loss = loss_CAPE + loss_BCE
        #loss = self.FCL(y_hat[:, :, self.mask == 1], y_p[:, :, self.mask == 1]) ##focal loss
        #loss = self.ERL(y_hat[:, :, self.mask == 1], y_p[:, :, self.mask == 1]) ##entropy reg loss
        self.train_losses.append([self.current_epoch, loss.item()])
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/BCE", loss_BCE)
        self.log("train/CAPE", loss_CAPE)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, ev_date = batch["x"], batch["y"], batch.get("ev_date")
        y_hat, y_hat_prob = self.forward(x, ev_date)
        y_p = []
        for i in range(len(self.thresholds)):
            y_p.append(y.gt(self.thresholds[i]).float())
        y_p = torch.cat(y_p, dim=1).to(self.device)
        loss = self.BCEL(y_hat[:, :, self.mask == 1], y_p[:, :, self.mask == 1])
        self.val_losses.append([self.current_epoch, loss.item()])
        self.log("val/loss", loss, prog_bar=True)

        metrics = dict()

        for metric in self.metrics:
            metrics[metric.__name__] = 0
            for j, th in enumerate(self.thresholds):
                if (
                    metric.__name__ == "AUC" and 1 not in y_p[:, j, self.mask == 1]
                ):  # if no pixel with more than 150mm can't calculate AUC, 1 by default
                    met = 1

                else:
                    met = metric(
                        y_p[:, j, self.mask == 1], y_hat_prob[:, j, self.mask == 1]
                    )
                metrics[metric.__name__] = metrics[metric.__name__] + met
                self.log(
                    f"val_metric/{metric.__name__} {self.thtot[j]:.0f}", met
                )

        for metric in self.metrics:  # print mean for metrics
            met = metrics[metric.__name__] / len(self.thresholds)
            self.log(f"val/{metric.__name__}", met)

    def test_step(self, batch, batch_idx):
        x, y, ev_date = batch["x"], batch["y"], batch.get("ev_date")
        y_hat, y_hat_prob = self.forward(x, ev_date)

        y_p = []
        for i in range(len(self.thresholds)):
            y_p.append(y.gt(self.thresholds[i]).float())
        y_p = torch.cat(y_p, dim=1)

        self.test_predictions.append(y_hat)

        metrics = dict()

        for metric in self.metrics:
            metrics[metric.__name__] = 0
            for j, th in enumerate(self.thresholds):
                if (
                    metric.__name__ == "AUC" and 1 not in y_p[:, j, self.mask == 1]
                ):  # if no pixel with more than 150mm can't calculate AUC, 1 by default
                    met = 1

                else:
                    met = metric(
                        y_p[:, j, self.mask == 1], y_hat_prob[:, j, self.mask == 1]
                    )
                metrics[metric.__name__] = metrics[metric.__name__] + met
                self.log(
                    f"test_metric/{metric.__name__} {self.thtot[j]:.0f}", met
                )

        for metric in self.metrics:  # print mean for metrics
            met = metrics[metric.__name__] / len(self.thresholds)
            self.log(f"test/{metric.__name__}", met)

        #Calculating Brier score input model
        input_models_brier_score = {}
        lv_thresholds = [1, 5, 10, 20, 50, 100, 150]
        y = y * self.case_study_max
        x = x * self.case_study_max
        for j, lv in enumerate(lv_thresholds):
            prob_input_models = (x > lv).float()
            input_models_brier_score[lv] = (
                ((prob_input_models - y.gt(lv).float()) ** 2).mean().item()
            )

            print(f">Brier score for input models {lv} mm: {input_models_brier_score[lv]:.4f}")

        # sns.set_style("whitegrid")

        # plt.figure()
        # plt.plot(
        #     lv_thresholds,
        #     [brier_scores[lv] for lv in lv_thresholds],
        #     label="Brier score",
        # )
        # plt.plot(
        #     lv_thresholds,
        #     [input_models_brier_score[lv] for lv in lv_thresholds],
        #     label="Input models Brier score",
        # )
        # plt.xlabel("Threshold (mm)")
        # plt.ylabel("Brier score")
        # plt.legend()
        # plt.savefig("brier_scores.png")
        # plt.savefig(Path(self.logger.log_dir)/"brier_scores.png")

    def log_images(self, features, masks, logits_, batch_idx):
        respath = Path(self.logger.log_dir) / "test_images"
        respath.mkdir(exist_ok=True)
        tensor_to_img = lambda t: (t.detach().cpu().numpy()[0] * 255)
        for img_idx, (image, y_true, y_pred) in enumerate(
            zip(features, masks, logits_)
        ):
            Image.fromarray(tensor_to_img(y_pred)).convert("L").save(
                respath / f"{batch_idx}_{img_idx}_pred.png"
            )
            Image.fromarray(tensor_to_img(y_true)).convert("L").save(
                respath / f"{batch_idx}_{img_idx}_gt.png"
            )
            for i in range(image.shape[0]):
                Image.fromarray(tensor_to_img(image[i : i + 1])).convert("L").save(
                    respath / f"{batch_idx}_{img_idx}_m{i}.png"
                )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, factor=0.5
                ),
                "monitor": "val/loss",
                "frequency": 1,
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }


def ECE(gt, probs):
    y_true_gt = gt.squeeze().cpu().flatten()
    probs = probs.squeeze().cpu().flatten()
    x_, y_ = calibration_curve(y_true_gt, probs, n_bins=10, strategy="quantile")
    ece = np.mean(np.abs(x_ - y_))
    return ece


def KL(gt, probs):
    eps = torch.Tensor([1e-10]).cuda()
    y_true_gt = gt.squeeze().flatten()
    probs = probs.squeeze().flatten()
    kl_prob_gt = -(
        probs * torch.log((y_true_gt + eps) / (probs + eps))
        + (1 - probs) * torch.log((1 - y_true_gt + eps) / (1 - probs + eps))
    ).mean()
    return kl_prob_gt


def AUC(gt, probs):
    # gt=gt.permute(0,2,3,1).reshape(gt.size(0)*gt.size(2)*gt.size(3),gt.size(1))
    return roc_auc_score(y_true=gt.flatten().cpu(), y_score=probs.flatten().cpu())


def brierScore(gt, probs):
    return ((probs - gt) ** 2).mean()


class EntropyRegularizedLoss(nn.Module):
    '''
    Loss regularized by entropy implementation
    L = CE - beta * H
    '''
    def __init__(self, beta=3, weight=None):
        super(EntropyRegularizedLoss, self).__init__()
        self.beta = beta
        self.weight = weight

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        """
        input: [N, C]
        target: [N, ]
        """
        #logpt = F.log_softmax(input, dim=1)
        logpt = torch.log(input)
        p_logp = (logpt * logpt.exp())
        entropy = -p_logp.sum(dim=1)
        loss = F.nll_loss(logpt, target) - self.beta * entropy.mean()
        return loss
    

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = torch.log(input)
        logpt = logpt.gather(1,int(target))
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: 
            return loss.mean()
        else: 
            return loss.sum()