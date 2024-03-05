from pathlib import Path
from utils import io
from PIL import Image

import torch
import torch.nn as nn
torch.set_float32_matmul_precision('high')
NUM_WORKERS = 0

from torch.utils.data import DataLoader, TensorDataset
from utils.datasets import NWPDataset
import segmentation_models_pytorch as smp
from networks.unet import UNet, ExtraUNet

import pytorch_lightning as pl

class SegmentationModel(pl.LightningModule):
	def __init__(self, **hparams):
		super().__init__()
		self.save_hyperparameters()

		self.load_data()
		if self.hparams.network_model == 'unet':
			self.cnn = UNet(self.in_features, self.out_features)
		elif self.hparams.network_model.startswith('extra'):
			self.cnn = ExtraUNet(self.in_features, self.out_features, image_shape=(self.x_train[0].shape[1], self.x_train[0].shape[2]), use_attention=True)
		else:
			raise NotImplementedError(f'Model {self.hparams.network_model} not implemented')
		self.loss = nn.MSELoss()

		self.rmse = lambda loss: (loss*(self.case_study_max**2)).sqrt().item()
		self.metrics = []
		self.test_predictions = []

		self.train_losses = []
		self.val_losses = []
		

	def forward(self, x, times):
		if isinstance(self.cnn, ExtraUNet):
			return self.cnn(x, times)
		return self.cnn(x)

	def load_data(self):
		case_study_max, available_models, train_dates, val_dates, test_dates, indices_one, indices_zero, mask, nx, ny = io.get_casestudy_stuff(
			self.hparams.input_path, self.hparams.split_idx, self.hparams.n_split, self.hparams.case_study, ispadded=False
		)
		self.x_train, self.y_train, in_features, out_features = io.load_data('unet', self.hparams.input_path, train_dates, case_study_max, indices_one, indices_zero, available_models)
		self.x_val, self.y_val, in_features, out_features = io.load_data('unet', self.hparams.input_path, val_dates, case_study_max, indices_one, indices_zero, available_models)
		self.x_test, self.y_test, in_features, out_features = io.load_data('unet', self.hparams.input_path, test_dates, case_study_max, indices_one, indices_zero, available_models)

		self.train_dates, self.val_dates, self.test_dates = train_dates, val_dates, test_dates
		self.case_study_max = case_study_max
		self.mask = torch.from_numpy(mask).float().to(self.device)
		self.in_features = in_features
		self.out_features = out_features
	
	def train_dataloader(self):
		if isinstance(self.cnn, ExtraUNet):
			train_dataset = NWPDataset((
				torch.from_numpy(self.x_train), torch.from_numpy(io.date_features(self.train_dates)),
				torch.from_numpy(self.y_train).unsqueeze(1)))
		else:
			train_dataset = NWPDataset((torch.from_numpy(self.x_train), torch.from_numpy(self.y_train).unsqueeze(1)))
		return DataLoader(train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=NUM_WORKERS)
		
	def val_dataloader(self):
		if isinstance(self.cnn, ExtraUNet):
			val_dataset = NWPDataset((
				torch.from_numpy(self.x_val), torch.from_numpy(io.date_features(self.val_dates)),
				torch.from_numpy(self.y_val).unsqueeze(1)))
		else:
			val_dataset = NWPDataset((torch.from_numpy(self.x_val), torch.from_numpy(self.y_val).unsqueeze(1)))
		return DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=NUM_WORKERS)
	
	def test_dataloader(self):
		if isinstance(self.cnn, ExtraUNet):
			test_dataset = NWPDataset((
				torch.from_numpy(self.x_test), torch.from_numpy(io.date_features(self.test_dates)),
				torch.from_numpy(self.y_test).unsqueeze(1)))
		else:
			test_dataset = NWPDataset((torch.from_numpy(self.x_test), torch.from_numpy(self.y_test).unsqueeze(1)))
		return DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=NUM_WORKERS)
	
	def training_step(self, batch, batch_idx):
		if not isinstance(self.cnn, ExtraUNet):
			x, y = batch
			# y = y * self.mask
			times = None
		else:
			x, times, y = batch
		y_hat = self.forward(x, times)
		loss = self.loss(y_hat, y)
		self.train_losses.append([self.current_epoch, loss.item()])
		self.log("train_loss", loss)
		self.log("train_rmse", self.rmse(loss), prog_bar=True)

		return loss

	def validation_step(self, batch, batch_idx):
		if not isinstance(self.cnn, ExtraUNet):
			x, y = batch
			times = None
			# y = y * self.mask
		else:
			x, times, y = batch
		y_hat = self.forward(x, times)
		loss = self.loss(y_hat, y)
		self.val_losses.append([self.current_epoch, loss.item()])
		self.log("val_loss", loss)
		self.log("val_rmse", self.rmse(loss), prog_bar=True)

		for metric in self.metrics:
			self.log(f"val_{metric.__name__}", metric(y_hat, y))
	
	def test_step(self, batch, batch_idx):
		if not isinstance(self.cnn, ExtraUNet):
			x, y = batch
			times = None
			# y = y * self.mask
		else:
			x, times, y = batch
		y_hat = self.forward(x, times)
		loss = self.loss(y_hat, y)
		self.log("test rmse", self.rmse(loss))
		# self.log_images(x, y, y_hat, batch_idx)

		self.test_predictions.append(y_hat)

		for channel in range(x.shape[1]):
			loss_ch = self.loss(x[:, channel:channel+1, :, :], y)
			self.log(f"rmse NWP {channel}", self.rmse(loss_ch))

		for metric in self.metrics:
			self.log(f"test_{metric.__name__}", metric(y_hat, y))
	
	def on_train_end(self):
		import seaborn as sns
		import pandas as pd
		import matplotlib.pyplot as plt
		fig, ax = plt.subplots()
		train_losses = pd.DataFrame(self.train_losses, columns=['epoch', 'loss'])
		val_losses = pd.DataFrame(self.val_losses, columns=['epoch', 'loss'])
		sns.lineplot(data=train_losses, x='epoch', y='loss', label='train')
		sns.lineplot(data=val_losses, x='epoch', y='loss', label='valid')
		plt.legend()
		plt.yscale('log')
		fig.savefig(Path(self.logger.log_dir)/"losses.png")


	def log_images(self, features, masks, logits_, batch_idx):
		respath = Path(self.logger.log_dir) / "test_images"
		respath.mkdir(exist_ok=True)
		tensor_to_img = lambda t: (t.detach().cpu().numpy()[0] * 255)
		for img_idx, (image, y_true, y_pred) in enumerate(zip(features, masks, logits_)):
			Image.fromarray(tensor_to_img(y_pred)).convert('L').save(respath/f"{batch_idx}_{img_idx}_pred.png")
			Image.fromarray(tensor_to_img(y_true)).convert('L').save(respath/f"{batch_idx}_{img_idx}_gt.png")
			for i in range(image.shape[0]):
				Image.fromarray(tensor_to_img(image[i:i+1])).convert('L').save(respath/f"{batch_idx}_{img_idx}_m{i}.png")

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
