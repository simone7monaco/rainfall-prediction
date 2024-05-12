from pathlib import Path
from utils import io
from PIL import Image
import pandas as pd
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.calibration import calibration_curve
import scipy

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_float32_matmul_precision("high")
NUM_WORKERS = 64

from torch.utils.data import DataLoader
from utils.datasets import NWPDataset
from networks.unet import UNet, ExtraUNet
from networks.vgunet import VGUNet

import pytorch_lightning as pl


class SegmentationModel(pl.LightningModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()

        self.load_data()
        if self.hparams.network_model == "unet":
            self.cnn = UNet(
                self.in_features, self.out_features, dropout=self.hparams.mcdropout
            )
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
        self.loss = nn.MSELoss()
        self.training_loss = nn.L1Loss()  # MSLELoss()
        self.brierLoss = BrierLoss()
        self.sigmoid = nn.Sigmoid()
        self.BCEL = nn.BCEWithLogitsLoss()
        self.BCE = nn.BCELoss()
        # self.loss = lambda y_hat, y: F.mse_loss(y_hat * self.mask, y * self.mask)

        self.rmse = lambda loss: (loss * (self.case_study_max**2)).sqrt().item()
        self.thresholds = [
            1 / self.case_study_max,
            5 / self.case_study_max,
            10 / self.case_study_max,
            20 / self.case_study_max,
            50 / self.case_study_max,
            100 / self.case_study_max,
            150 / self.case_study_max,
        ]
        self.metrics = []  # [freqbias, ets, csi]
        self.test_predictions = []

        self.sigma = 0.1
        self.window = 500

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
        self.out_features = 7  # out_features

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
        loss1 = 0
        if self.hparams.fine_tune == 1:  # and self.current_epoch %2==0):
            n_bins = 100
            if (
                self.hparams.finetune_type == "bin"
                or self.hparams.finetune_type == "kde"
            ):
                # sort to calculate bins
                y_hat_prob_mask = y_hat_prob[:, :, self.mask == 1].flatten()
                sorted_idx = torch.argsort(y_hat_prob_mask)  ########
                targets_probs = y_hat_prob_mask[sorted_idx]
                labels = y_p[:, :, self.mask == 1].flatten()
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
                    sigma = self.sigma
                    for i in range(num_sample):
                        left = np.maximum(0, i - self.window)
                        right = np.minimum(i + self.window, num_sample)
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
                loss1 = self.BCE(targets_probs, new_labels)
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
                loss1 = self.BCEL(
                    y_hat[:, :, self.mask == 1], probs_emp[:, :, self.mask == 1]
                )
            else:
                raise NotImplementedError

        # else:
        loss2 = self.BCEL(y_hat[:, :, self.mask == 1], y_p[:, :, self.mask == 1])
        loss = 0.3 * loss1 + loss2
        self.train_losses.append([self.current_epoch, loss.item()])
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, ev_date = batch["x"], batch["y"], batch.get("ev_date")
        y_hat, y_hat_prob = self.forward(x, ev_date)
        y_p = []
        for i in range(len(self.thresholds)):
            y_p.append(y.gt(self.thresholds[i]).float())
        y_p = torch.cat(y_p, dim=1)
        loss = self.BCEL(y_hat, y_p)
        self.val_losses.append([self.current_epoch, loss.item()])
        self.log("val_loss", loss, prog_bar=True)

        if self.current_epoch % 10 == 0:
            for metric in self.metrics:
                for th in self.thresholds:
                    self.log(
                        f"val_{metric.__name__}_{th*self.case_study_max}",
                        metric(
                            y_hat[:, :, self.mask == 1], y[:, :, self.mask == 1], th
                        ),
                    )

    def test_step(self, batch, batch_idx):
        x, y, ev_date = batch["x"], batch["y"], batch.get("ev_date")
        y_hat, y_hat_prob = self.forward(x, ev_date)

        self.test_predictions.append(y_hat)

        for channel in range(x.shape[1]):
            loss_ch = self.loss(x[:, channel : channel + 1, :, :], y)
            self.log(f"rmse NWP {channel}", self.rmse(loss_ch))

        for metric in self.metrics:
            for th in self.thresholds:
                self.log(
                    f"test_{metric.__name__}_{th*self.case_study_max}",
                    metric(y_hat[:, :, self.mask == 1], y[:, :, self.mask == 1], th),
                )

        # Calculating Brier score
        save_dir = Path("proba")
        brier_scores = {}
        input_models_brier_score = {}
        lv_thresholds = [1, 5, 10, 20, 50, 100, 150]
        y = y * self.case_study_max
        x = x * self.case_study_max
        for j, lv in enumerate(lv_thresholds):
            (save_dir / str(lv)).mkdir(exist_ok=True)
            # save the probability results for each threshold under the logger directory / <threshold> / pred{ev_date}.csv

            for i, ename in enumerate(ev_date):
                # probabilities[lv] has shape (n_samples, C, H, W) to get the current image, we need to index the first dimension
                # img = probabilities[lv][i].cpu().numpy()
                pd.DataFrame(y_hat_prob[i][j].squeeze().cpu().numpy()).to_csv(
                    save_dir / f"{lv}/pred{ename}.csv", index=False, header=False
                )

            brier_scores[lv] = (
                ((y_hat_prob[:, j] - y.cuda().gt(lv).float()) ** 2).mean().item()
            )
            brier_scores[lv] = (
                brier_scores[lv] * (96 * 128) / 5247
            )  # normalization to mask==1 only
            prob_input_models = (x > lv).float()

            ece = ECE(gt=y.gt(lv).float(), probs=y_hat_prob[:, j], self=self)
            kl = KL(gt=y.gt(lv).float(), probs=y_hat_prob[:, j], self=self)
            input_models_brier_score[lv] = (
                ((prob_input_models - y.gt(lv).float()) ** 2).mean().item()
            )

            print(f"Brier score for threshold {lv} mm: {brier_scores[lv]:.4f}")
            print(f">Brier score for input models: {input_models_brier_score[lv]:.4f}")
            print(f"ECE for threshold {lv} mm: {ece:.4f}")
            print(f"KL for threshold {lv} mm: {kl:.4f}\n")
            self.log(f"Brier score {lv} mm", brier_scores[lv])
            self.log(f"ECE {lv} mm", ece)
            self.log(f"KL {lv} mm", kl)

        sns.set_style("whitegrid")

        plt.figure()
        plt.plot(
            lv_thresholds,
            [brier_scores[lv] for lv in lv_thresholds],
            label="Brier score",
        )
        plt.plot(
            lv_thresholds,
            [input_models_brier_score[lv] for lv in lv_thresholds],
            label="Input models Brier score",
        )
        plt.xlabel("Threshold (mm)")
        plt.ylabel("Brier score")
        plt.legend()
        plt.savefig("brier_scores.png")
        # plt.savefig(Path(self.logger.log_dir)/"brier_scores.png")

    # def on_train_end(self):
    # 	import seaborn as sns
    # 	import pandas as pd
    # 	import matplotlib.pyplot as plt
    # 	fig, ax = plt.subplots()
    # 	train_losses = pd.DataFrame(self.train_losses, columns=['epoch', 'loss'])
    # 	val_losses = pd.DataFrame(self.val_losses, columns=['epoch', 'loss'])
    # 	sns.lineplot(data=train_losses, x='epoch', y='loss', label='train')
    # 	sns.lineplot(data=val_losses, x='epoch', y='loss', label='valid')
    # 	plt.legend()
    # 	plt.yscale('log')
    # 	fig.savefig(Path(self.logger.log_dir)/"losses.png")

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
                    optimizer, factor=0.8
                ),
                "monitor": "val_loss",
                "frequency": "1",
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

    def get_monte_carlo_predictions(self, forward_passes=20, save_dir=None):
        """Function to get the monte-carlo samples and uncertainty estimates
        through multiple forward passes

        Parameters
        ----------
        data_loader : object
                data loader object from the data loader module
        forward_passes : int
                number of monte-carlo samples/forward passes
        """
        assert hasattr(self.cnn, "eval_dp"), "Model does not have dropout layers"

        dropout_predictions = []
        for i in range(forward_passes):
            predictions = []
            self.cnn.eval_dp()
            with torch.no_grad():
                for batch in self.test_dataloader():
                    x, y, ev_date = batch["x"], batch["y"], batch.get("ev_date")
                    x = x.to("cuda")
                    output = self.cnn(x) * self.mask.cuda()
                    predictions.append(output)
                predictions = torch.cat(
                    predictions, dim=0
                )  # shape (n_samples, C, H, W)
            dropout_predictions.append(predictions)
        dropout_predictions = torch.stack(
            dropout_predictions, dim=0
        )  # shape (n_forward_passes, n_samples, C, H, W)
        y_all = (
            torch.cat([batch["y"] for batch in self.test_dataloader()], dim=0)
            * self.mask.cpu()
        )

        # Calculating stats across multiple MCD forward passes
        mean = dropout_predictions.mean(dim=0)
        variance = dropout_predictions.var(dim=0)

        # Calculating variance over error
        error = torch.abs(mean - y_all.cuda())
        error = error.flatten().cpu().numpy() * self.case_study_max
        variance = variance.flatten().cpu().numpy()

        sns.set_style("whitegrid")

        if save_dir is None:
            save_dir = Path(self.logger.log_dir)
        save_dir.mkdir(exist_ok=True)

        ind = np.where(error > 0)
        var_mean = []
        err_mean = []
        for i, bin in enumerate(np.linspace(0, 400, 201)):
            indx = np.where((error > bin) & (error <= bin + (400 / 200)))
            var_mean.append(variance[indx].mean())
            err_mean.append(error[indx].mean())

        plt.figure()
        plt.scatter(err_mean, var_mean)
        plt.xlabel("Prediction error (mm)")
        plt.ylabel("variance")
        plt.savefig(
            save_dir
            / f"error_variance_seed_{self.hparams.seed}_split_{self.hparams.n_split}.png"
        )

        plt.figure()
        plt.hist(error[ind], bins=np.linspace(0, 20, 100))
        plt.xlabel("Prediction error (mm)")
        plt.ylabel("# pixel")
        plt.savefig(
            save_dir
            / f"pred_error_seed_{self.hparams.seed}_split_{self.hparams.n_split}.png"
        )

        plt.figure()
        plt.hist(error[ind], bins=100, log=True)
        plt.xlabel("Prediction error (mm)")
        plt.ylabel("log(# pixel)")
        plt.savefig(
            save_dir
            / f"pred_error_log_seed_{self.hparams.seed}_split_{self.hparams.n_split}.png"
        )

        # Calculating entropy across multiple MCD forward passes
        # entropy = -torch.sum(mean * torch.log(mean + 1e-6), axis=-1)

        # # Calculating mutual information across multiple MCD forward passes
        # mutual_info = entropy - torch.mean(torch.sum(-dropout_predictions * torch.log(dropout_predictions + 1e-6),
        # 									dim=-1), dim=0)

        y_all = y_all.cuda()
        loss = self.loss(mean, y_all)
        # print(f"y_all shape {y_all.shape}")
        # print(f"mean shape {mean.shape}")

        print(f"MCD RMSE", self.rmse(loss))
        print(f"MCD variance", variance.mean().item())
        print(f"forward pass ", forward_passes)
        wandb.log({"test rmse": self.rmse(loss)})
        wandb.log({"test variance": variance.mean().item()})
        # print(f"MCD entropy", entropy.mean().item())
        # print(f"MCD mutual info", mutual_info.mean().item())

        for metric in self.metrics:
            for th in self.thresholds:
                met = metric(
                    mean[:, :, self.mask == 1], y_all[:, :, self.mask == 1], th
                ).item()
                wandb.log({f"MCD_{metric.__name__}_{th*self.case_study_max}": met})
                print(f"MCD_{metric.__name__}_{th*self.case_study_max}: {met:.4f}")

    def eval_proba(
        self,
        lv_thresholds=[1, 5, 10, 20, 50, 100, 150],
        forward_passes=20,
        save_dir=None,
    ):
        """
        Perform an evaluation loop for the probabilistic forecasts. The function have to store, for each threshold and each image,
        the probability for the models (the MCD samples) to be above the threshold. The function will then compute the Brier score
        """

        assert hasattr(self.cnn, "eval_dp"), "Model does not have dropout layers"

        if save_dir is None:
            save_dir = Path(self.logger.log_dir)
        save_dir.mkdir(exist_ok=True)

        probabilities = {lv: [] for lv in lv_thresholds}
        y_all = (
            torch.cat([batch["y"] for batch in self.test_dataloader()], dim=0)
            * self.case_study_max
        )
        enames_all = torch.cat(
            [batch["ev_date"] for batch in self.test_dataloader()], dim=0
        )

        self.cnn.eval_dp()
        with torch.no_grad():
            for i in range(forward_passes):
                predictions = []
                for batch in self.test_dataloader():
                    x = batch["x"].cuda()
                    output = self.cnn(x) * self.mask.cuda()
                    predictions.append(output)
                predictions = torch.cat(predictions, dim=0)
                predictions = predictions * self.case_study_max
                for lv in lv_thresholds:
                    probabilities[lv].append((predictions > lv).float())
            for lv in lv_thresholds:
                probabilities[lv] = torch.stack(probabilities[lv], dim=0).mean(dim=0)

        # Calculating Brier score
        brier_scores = {}
        input_models_brier_score = {}
        x_all = (
            torch.cat([batch["x"] for batch in self.test_dataloader()], dim=0)
            * self.case_study_max
        )
        for lv in lv_thresholds:
            (save_dir / str(lv)).mkdir(exist_ok=True)
            # save the probability results for each threshold under the logger directory / <threshold> / pred{ev_date}.csv

            for i, ename in enumerate(enames_all):
                # probabilities[lv] has shape (n_samples, C, H, W) to get the current image, we need to index the first dimension
                # img = probabilities[lv][i].cpu().numpy()
                pd.DataFrame(probabilities[lv][i].squeeze().cpu().numpy()).to_csv(
                    save_dir / f"{lv}/pred{ename}.csv", index=False, header=False
                )

            brier_scores[lv] = (
                ((probabilities[lv] - y_all.cuda().gt(lv).float()) ** 2).mean().item()
            )
            brier_scores[lv] = (
                brier_scores[lv] * (96 * 128) / 5247
            )  # normalization to mask==1 only
            prob_input_models = (x_all > lv).float()
            # print(f"y_all shape {y_all.shape}")
            # print(f"input_model_all shape {x_all.shape}")
            # print(f"probabilities shape {probabilities[lv].shape}")
            # print(f"prob_input_models shape {prob_input_models.shape}")
            # print(f"diff shape {(prob_input_models - y_all.gt(lv).float()).shape}")

            ece = ECE(gt=y_all.gt(lv).float(), probs=probabilities[lv], self=self)
            kl = KL(gt=y_all.gt(lv).float(), probs=probabilities[lv], self=self)
            input_models_brier_score[lv] = (
                ((prob_input_models - y_all.gt(lv).float()) ** 2).mean().item()
            )

            print(f"Brier score for threshold {lv} mm: {brier_scores[lv]:.4f}")
            print(f">Brier score for input models: {input_models_brier_score[lv]:.4f}")
            print(f"ECE for threshold {lv} mm: {ece:.4f}")
            print(f"KL for threshold {lv} mm: {kl:.4f}\n")
            wandb.log({f"Brier score {lv} mm": brier_scores[lv]})

        sns.set_style("whitegrid")

        plt.figure()
        plt.plot(
            lv_thresholds,
            [brier_scores[lv] for lv in lv_thresholds],
            label="Brier score",
        )
        plt.plot(
            lv_thresholds,
            [input_models_brier_score[lv] for lv in lv_thresholds],
            label="Input models Brier score",
        )
        plt.xlabel("Threshold (mm)")
        plt.ylabel("Brier score")
        plt.legend()
        plt.savefig("brier_scores.png")
        # plt.savefig(Path(self.logger.log_dir)/"brier_scores.png")


class BrierLoss(nn.Module):
    def __init__(self):
        super(BrierLoss, self).__init__()

        self.case_study_max = 483.717752
        self.lv_thresholds = [
            1 / self.case_study_max,
            5 / self.case_study_max,
            10 / self.case_study_max,
            20 / self.case_study_max,
            50 / self.case_study_max,
            100 / self.case_study_max,
            150 / self.case_study_max,
        ]
        # self.lv_thresholds=[50/self.case_study_max]

    def forward(self, predictions, targets):
        brier_score = torch.tensor(0.0, requires_grad=True)
        for lv in self.lv_thresholds:
            brier_score = (
                brier_score
                + ((predictions[lv].float() - targets.gt(lv).float()) ** 2).mean()
            )
        return brier_score


def freqbias(perc, veri, threshh):
    hits = torch.sum((veri >= threshh) * (perc >= threshh))
    falsealarms = torch.sum((veri < threshh) * (perc >= threshh))
    misses = torch.sum((veri >= threshh) * (perc < threshh))
    return (hits + falsealarms) / (hits + misses)


def ets(perc, veri, threshh):
    hits = torch.sum((veri >= threshh) * (perc >= threshh))
    falsealarms = torch.sum((veri < threshh) * (perc >= threshh))
    misses = torch.sum((veri >= threshh) * (perc < threshh))
    correctnegatives = torch.sum((veri < threshh) * (perc < threshh))
    hitsrandom = (
        (hits + misses)
        * (hits + falsealarms)
        / (hits + falsealarms + misses + correctnegatives)
    )
    return (hits - hitsrandom) / (hits + misses + falsealarms - hitsrandom)


def csi(perc, veri, threshh):
    hits = torch.sum((veri >= threshh) * (perc >= threshh))
    falsealarms = torch.sum((veri < threshh) * (perc >= threshh))
    misses = torch.sum((veri >= threshh) * (perc < threshh))
    return hits / (hits + falsealarms + misses)


def ECE(gt, probs, self):
    gt = gt.squeeze().cpu()
    probs = probs.squeeze().cpu()
    probs = probs[:, self.mask.cpu() == 1].flatten()
    y_true_gt = gt[:, self.mask.cpu() == 1].flatten()
    x_, y_ = calibration_curve(y_true_gt, probs, n_bins=10, strategy="quantile")
    ece = np.mean(np.abs(x_ - y_))
    return ece


def KL(gt, probs, self):
    eps = torch.Tensor([1e-10]).cpu()
    gt = gt.squeeze()
    probs = probs.squeeze()
    probs = probs[:, self.mask.cpu() == 1].flatten().cpu()
    y_true_gt = gt[:, self.mask.cpu() == 1].flatten().cpu()
    kl_prob_gt = -(
        probs * torch.log((y_true_gt + eps) / (probs + eps))
        + (1 - probs) * torch.log((1 - y_true_gt + eps) / (1 - probs + eps))
    ).mean()
    return kl_prob_gt
