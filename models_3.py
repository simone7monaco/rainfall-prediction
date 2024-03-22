from pathlib import Path
from utils import io
from PIL import Image

import torch
import torch.nn as nn
import numpy as np
torch.set_float32_matmul_precision('high')
NUM_WORKERS = 32

from torch.utils.data import DataLoader, TensorDataset
from utils.datasets import NWPDataset
import segmentation_models_pytorch as smp
from networks.unet import UNet, ExtraUNet

import pytorch_lightning as pl

def iou(pred_, gt):
	intersection = pred_.int() & gt.int()
	union = pred_.int() | gt.int()
	return intersection.sum() / union.sum()
    

class SegmentationModel(pl.LightningModule):
	def __init__(self, **hparams):
		super().__init__()
		self.save_hyperparameters()

		self.load_data()
		if self.hparams.network_model == 'unet':
			self.cnn = UNet(self.in_features, self.out_features)
		elif self.hparams.network_model.startswith('extra'):
			self.cnn = ExtraUNet(self.in_features, self.out_features, image_shape=(self.x_train[0].shape[1], self.x_train[0].shape[2]), use_attention=True)
		elif self.hparams.network_model == 'unet_2':
			self.cnn = UNet(self.in_features, 1)
			self.cnn_1 = UNet(self.in_features, self.out_features)
		else:
			raise NotImplementedError(f'Model {self.hparams.network_model} not implemented')
		self.loss = nn.MSELoss()
		self.loss_segm = nn.BCEWithLogitsLoss()
		self.lossL1 = nn.L1Loss()

		self.rmse = lambda loss: (loss*(self.case_study_max**2)).sqrt().item()
		if self.hparams.network_model == 'unet_2':
			self.metrics = [iou]
		else:
			self.metrics = []
		self.test_predictions = []

		self.train_losses = []
		self.val_losses = []

	def forward(self, x, times):
		if isinstance(self.cnn, ExtraUNet):
			return None, self.cnn(x, times)
		if self.hparams.network_model == 'unet_2':
			x_logits_segmentation = self.cnn(x)*self.mask
			x_segmentation = torch.round(torch.sigmoid(x_logits_segmentation))
			x2 = x * x_segmentation
			return x_logits_segmentation, self.cnn_1(x2)*x_segmentation
		return None, self.cnn(x)*self.mask # * torch.heaviside(y, torch.tensor([0]).float().to(self.device)))* torch.heaviside(y, torch.tensor([0]).float().to(self.device)) #mod

	def load_data(self):
		case_study_max, available_models, train_dates, val_dates, test_dates, indices_one, indices_zero, mask, nx, ny = io.get_casestudy_stuff(
			self.hparams.input_path, self.hparams.split_idx, self.hparams.n_split, self.hparams.case_study, ispadded=True) #was False
		self.x_train, self.y_train, in_features, out_features = io.load_data('unet', self.hparams.input_path, train_dates, case_study_max, indices_one, indices_zero, available_models)
		self.x_val, self.y_val, in_features, out_features = io.load_data('unet', self.hparams.input_path, val_dates, case_study_max, indices_one, indices_zero, available_models)
		self.x_test, self.y_test, in_features, out_features = io.load_data('unet', self.hparams.input_path, test_dates, case_study_max, indices_one, indices_zero, available_models)

		self.train_dates, self.val_dates, self.test_dates = train_dates, val_dates, test_dates
		self.case_study_max = case_study_max
		self.mask = torch.from_numpy(mask).float().to('cuda') #.to(self.device)
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
		#y_segm = torch.heaviside(y, torch.tensor([0]).float().to(self.device))
		y_segm = torch.where(y>0.001, 1, 0).float()
		y_hat_segm, y_hat = self.forward(x, times) #mod
		loss = self.lossL1(y_hat, y)
		if self.hparams.network_model == 'unet_2':
			loss_segm = self.loss_segm(y_hat_segm, y_segm) /100
		else:
			loss_segm = torch.Tensor([0]).to(self.device)
		self.train_losses.append([self.current_epoch, loss.item(), loss_segm.item()])
		self.log("train_loss_regr", loss)
		self.log("train_loss_segm", loss_segm)
		self.log("train_loss", loss+loss_segm)
		self.log("train_rmse", self.rmse(loss), prog_bar=True)

		return loss + loss_segm

	def validation_step(self, batch, batch_idx):
		if not isinstance(self.cnn, ExtraUNet):
			x, y = batch
			times = None
			# y = y * self.mask
		else:
			x, times, y = batch
		#y_segm = torch.heaviside(y, torch.tensor([0]).float().to(self.device))
		y_segm = torch.where(y>0.001, 1, 0).float()
		y_hat_segm, y_hat = self.forward(x, times) #mod
		loss = self.loss(y_hat, y)
		if self.hparams.network_model == 'unet_2':
			loss_segm = self.loss_segm(y_hat_segm, y_segm) /100
		else:
			loss_segm = torch.Tensor([0]).to(self.device)
		self.val_losses.append([self.current_epoch, loss.item(), loss_segm.item()])
		self.log("train_loss_regr", loss)
		self.log("val_loss_segm", loss_segm)
		self.log("val_loss", loss+loss_segm)
		self.log("val_rmse", self.rmse(loss), prog_bar=True)

		for metric in self.metrics:
			self.log(f"val_{metric.__name__}", metric(y_hat_segm, y_segm))
	
	def test_step(self, batch, batch_idx):
		if not isinstance(self.cnn, ExtraUNet):
			x, y = batch
			times = None
			# y = y * self.mask
		else:
			x, times, y = batch
		y_hat_segm, y_hat = self.forward(x, times)
		y_segm = torch.where(y>0.001, 1, 0).float()
		loss = self.loss(y_hat*self.mask, y*self.mask)
		self.log("test rmse", self.rmse(loss))
		#self.log_images(x, y, y_hat, batch_idx)
		print(f"sum pred: {y_hat.sum()}, sum pred*mask: {(y_hat*self.mask).sum()}, sum y: {y.sum()}, sum y*mask: {(y*self.mask).sum()}")
		print(f"y shape: {y.squeeze().shape}, y_hat shape: {y_hat.squeeze().shape}, x0 shape: {x[:, 0].squeeze().shape}, x1 shape: {x[:, 1].squeeze().shape}, x2 shape: {x[:, 2].shape}, x3 shape: {x[:, 3].shape}")
		print(f"np.rmse: {np.sqrt(np.square(y.to('cpu').detach().numpy()*self.case_study_max-y_hat.squeeze().to('cpu').detach().numpy()*self.case_study_max).mean())}")
		print(f"torch loss: {(nn.MSELoss()(y_hat*self.case_study_max, y*self.case_study_max))**.5}")
		for i in range(4):
			print(torch.tensor(x[:, i]).sum())

		self.test_predictions.append(y_hat)

		for channel in range(x.shape[1]):
			loss_ch = self.loss(x[:, channel:channel+1, :, :], y)
			self.log(f"rmse NWP {channel}", self.rmse(loss_ch))

		for metric in self.metrics:
			self.log(f"test_{metric.__name__}", metric(y_hat_segm, y_segm))
	
	def on_train_end(self):
		import seaborn as sns
		import pandas as pd
		import matplotlib.pyplot as plt
		fig, ax = plt.subplots()
		train_losses = pd.DataFrame(self.train_losses, columns=['epoch', 'loss', 'loss_segm'])
		val_losses = pd.DataFrame(self.val_losses, columns=['epoch', 'loss', 'loss_segm'])
		sns.lineplot(data=train_losses, x='epoch', y='loss', label='train')
		sns.lineplot(data=val_losses, x='epoch', y='loss', label='valid')
		plt.legend()
		plt.yscale('log')
		fig.savefig(Path(self.hparams.output_path)/"losses.png")


	def log_images(self, features, masks, logits_, batch_idx):
		respath = Path(self.hparams.output_path) / "test_images"
		respath.mkdir(exist_ok=True)
		tensor_to_img = lambda t: (t.detach().cpu().numpy()[0] * 255)
		for img_idx, (image, y_true, y_pred) in enumerate(zip(features, masks, logits_)):
			Image.fromarray(tensor_to_img(y_pred)).convert('L').save(respath/f"{batch_idx}_{img_idx}_pred.png")
			Image.fromarray(tensor_to_img(y_true)).convert('L').save(respath/f"{batch_idx}_{img_idx}_gt.png")
			for i in range(image.shape[0]):
				Image.fromarray(tensor_to_img(image[i:i+1])).convert('L').save(respath/f"{batch_idx}_{img_idx}_m{i}.png")

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

	#def configure_optimizers(self):
		#optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
		#return {
			#"optimizer": optimizer,
			#"lr_scheduler": {
				#"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
				#"interval": "epoch",
				#"monitor": "val_loss",
				#"frequency": "1",
			#},
		#}
  
  
class SegmentationModel_1(pl.LightningModule):
	def __init__(self, **hparams):
		super().__init__()
		self.save_hyperparameters()

		self.load_data()
		self.cnn = UNet(self.in_features, 3)
		self.loss = nn.BCELoss()

		self.rmse = lambda loss: (loss*(self.case_study_max**2)).sqrt().item()
		self.metrics = [iou]
		self.test_predictions = []

		self.train_losses = []
		self.val_losses = []

	def forward(self, x):
		x_s = self.cnn(x)
		print(f"x_s shape: {x_s.shape}")
		return x_s

	def load_data(self):
		case_study_max, available_models, train_dates, val_dates, test_dates, indices_one, indices_zero, mask, nx, ny = io.get_casestudy_stuff(
			self.hparams.input_path, self.hparams.split_idx, self.hparams.n_split, self.hparams.case_study, ispadded=True) #was False
		self.x_train, self.y_train, in_features, out_features = io.load_data('unet', self.hparams.input_path, train_dates, case_study_max, indices_one, indices_zero, available_models)
		self.x_val, self.y_val, in_features, out_features = io.load_data('unet', self.hparams.input_path, val_dates, case_study_max, indices_one, indices_zero, available_models)
		self.x_test, self.y_test, in_features, out_features = io.load_data('unet', self.hparams.input_path, test_dates, case_study_max, indices_one, indices_zero, available_models)

		self.train_dates, self.val_dates, self.test_dates = train_dates, val_dates, test_dates
		self.case_study_max = case_study_max
		self.mask = torch.from_numpy(mask).float().to('cuda') #.to(self.device)
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
		y_segm_H = torch.where(y>self.hparams.where_threshold_H, 1, 0).float()
		y_segm_LH = torch.where(y<=self.hparams.where_threshold_H and y>=self.hparams.where_threshold_L, 1, 0).float()
		y_segm_L = torch.where(y<self.hparams.where_threshold_L, 1, 0).float()
		y_segm = torch.cat(y_segm_L, y_segm_LH, y_segm_H)
		y_hat_segm = self.forward(x)
		print(f"y_segm shape: {y_segm.shape}")
		print(F"y_hat_segm shape: {y_hat_segm.shape}")
		loss_segm = self.loss(y_hat_segm, y_segm)
		self.train_losses.append([self.current_epoch, loss_segm.item()])
		self.log("train_loss_segm", loss_segm)

		return loss_segm

	def validation_step(self, batch, batch_idx):
		if not isinstance(self.cnn, ExtraUNet):
			x, y = batch
			times = None
			# y = y * self.mask
		else:
			x, times, y = batch
		y_segm_H = torch.where(y>self.hparams.where_threshold_H, 1, 0).float()
		y_segm_L = torch.where(y<self.hparams.where_threshold_L, 0, 1).float()
		y_segm = torch.where(y>((self.hparams.where_threshold_H+self.hparams.where_threshold_L)/2), 1, 0).float()
		y_hat_segm = self.forward(x)
		loss_segm_H = self.loss(y_hat_segm, y_segm_H)
		loss_segm_L = self.loss(y_hat_segm, y_segm_L)
		self.train_losses.append([self.current_epoch, loss_segm_H.item() + loss_segm_L.item()])
		self.log("train_loss_segm_H", loss_segm_H)
		self.log("train_loss_segm_L", loss_segm_L)

		for metric in self.metrics:
			self.log(f"val_{metric.__name__}", metric(y_hat_segm.gt(self.hparams.sigmoid_threshold), y_segm))
	
	def test_step(self, batch, batch_idx):
		if not isinstance(self.cnn, ExtraUNet):
			x, y = batch
			times = None
			# y = y * self.mask
		else:
			x, times, y = batch
		y_hat_segm = self.forward(x).gt(self.hparams.sigmoid_threshold)
		y_segm = torch.where(y>((self.hparams.where_threshold_H+self.hparams.where_threshold_L)/2), 1, 0).float()
		#self.log_images(x, y, y_hat, batch_idx)

		self.test_predictions.append(y_hat_segm)

		for metric in self.metrics:
			self.log(f"test_{metric.__name__}", metric(y_hat_segm, y_segm))
	
	def on_train_end(self):
		import seaborn as sns
		import pandas as pd
		import matplotlib.pyplot as plt
		fig, ax = plt.subplots()
		train_losses = pd.DataFrame(self.train_losses, columns=['epoch', 'loss_segm'])
		val_losses = pd.DataFrame(self.val_losses, columns=['epoch', 'loss_segm'])
		sns.lineplot(data=train_losses, x='epoch', y='loss_segm', label='train')
		sns.lineplot(data=val_losses, x='epoch', y='loss_segm', label='valid')
		plt.legend()
		plt.yscale('log')
		fig.savefig(Path(self.hparams.output_path)/"losses.png")


	def log_images(self, features, masks, logits_, batch_idx):
		respath = Path(self.hparams.output_path) / "test_images"
		respath.mkdir(exist_ok=True)
		tensor_to_img = lambda t: (t.detach().cpu().numpy()[0] * 255)
		for img_idx, (image, y_true, y_pred) in enumerate(zip(features, masks, logits_)):
			Image.fromarray(tensor_to_img(y_pred)).convert('L').save(respath/f"{batch_idx}_{img_idx}_pred.png")
			Image.fromarray(tensor_to_img(y_true)).convert('L').save(respath/f"{batch_idx}_{img_idx}_gt.png")
			for i in range(image.shape[0]):
				Image.fromarray(tensor_to_img(image[i:i+1])).convert('L').save(respath/f"{batch_idx}_{img_idx}_m{i}.png")

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class SegmentationModel_H(pl.LightningModule):
	def __init__(self, model_1_hparams,
                 model_1_params = None, **hparams):
		super().__init__()
		self.save_hyperparameters()

		self.load_data()
		if self.hparams.network_model == 'unet':
			self.cnn = UNet(self.in_features, self.out_features)
		elif self.hparams.network_model.startswith('extra'):
			self.cnn = ExtraUNet(self.in_features, self.out_features, image_shape=(self.x_train[0].shape[1], self.x_train[0].shape[2]), use_attention=True)
		elif self.hparams.network_model == 'unet_3':
			self.cnn = UNet(self.in_features, self.out_features)
			self.model_1 = SegmentationModel_1(**model_1_hparams)
			if model_1_params:
				self.model_1.load_state_dict(model_1_params)
			self.model_1.freeze()
			self.save_hyperparameters(ignore=['model_1'])
		else:
			raise NotImplementedError(f'Model {self.hparams.network_model} not implemented')
		self.loss = nn.MSELoss()
		self.lossL1 = nn.L1Loss()

		self.rmse = lambda loss: (loss*(self.case_study_max**2)).sqrt().item()
		self.metrics = []
		self.test_predictions = []

		self.train_losses = []
		self.val_losses = []
		

	def forward(self, x, times):
		if isinstance(self.cnn, ExtraUNet):
			return self.cnn(x, times)
		if self.hparams.network_model == 'unet_3':
			x_segmentation = self.model_1(x).gt(self.hparams.sigmoid_threshold)
			x2 = x * x_segmentation
			return self.cnn(x2)*x_segmentation
		return self.cnn(x)

	def load_data(self):
		case_study_max, available_models, train_dates, val_dates, test_dates, indices_one, indices_zero, mask, nx, ny = io.get_casestudy_stuff(
			self.hparams.input_path, self.hparams.split_idx, self.hparams.n_split, self.hparams.case_study, ispadded=True) #was False
		self.x_train, self.y_train, in_features, out_features = io.load_data('unet', self.hparams.input_path, train_dates, case_study_max, indices_one, indices_zero, available_models)
		self.x_val, self.y_val, in_features, out_features = io.load_data('unet', self.hparams.input_path, val_dates, case_study_max, indices_one, indices_zero, available_models)
		self.x_test, self.y_test, in_features, out_features = io.load_data('unet', self.hparams.input_path, test_dates, case_study_max, indices_one, indices_zero, available_models)

		self.train_dates, self.val_dates, self.test_dates = train_dates, val_dates, test_dates
		self.case_study_max = case_study_max
		self.mask = torch.from_numpy(mask).float().to('cuda') #.to(self.device)
		self.in_features = in_features
		self.out_features = out_features
		#print(f"nx: {nx}, ny: {ny}, sum: {mask.sum()}")
	
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
		loss = self.lossL1(y_hat, y)
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
		#self.log_images(x, y, y_hat, batch_idx)

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
		fig.savefig(Path(self.hparams.output_path)/"losses.png")


	def log_images(self, features, masks, logits_, batch_idx):
		respath = Path(self.hparams.output_path) / "test_images"
		respath.mkdir(exist_ok=True)
		tensor_to_img = lambda t: (t.detach().cpu().numpy()[0] * 255)
		for img_idx, (image, y_true, y_pred) in enumerate(zip(features, masks, logits_)):
			Image.fromarray(tensor_to_img(y_pred)).convert('L').save(respath/f"{batch_idx}_{img_idx}_pred.png")
			Image.fromarray(tensor_to_img(y_true)).convert('L').save(respath/f"{batch_idx}_{img_idx}_gt.png")
			for i in range(image.shape[0]):
				Image.fromarray(tensor_to_img(image[i:i+1])).convert('L').save(respath/f"{batch_idx}_{img_idx}_m{i}.png")

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
