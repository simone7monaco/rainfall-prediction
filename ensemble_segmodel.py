import torch
from torch import nn
from base_segmodel import _SegmentationModel
from models import UNet


class SegmentationModel(_SegmentationModel):
	"""
	Wrapper of the PL Model instantiating N instances of the UNet model and training them all together
	"""
	def __init__(self, **hparams):
		super().__init__(**hparams)

	def get_model(self):
		self.n_models = self.hparams.n_models if hasattr(self.hparams, 'n_models') else 5
		self.automatic_optimization = False
		return nn.ModuleList([UNet(self.in_features, self.out_features, dropout=self.hparams.mcdropout) for _ in range(self.n_models)])

	def configure_optimizers(self):
		return [torch.optim.Adam(self.cnn[i].parameters(), lr=self.hparams.lr) for i in range(self.n_models)]
	
	def forward(self, x, idx):
		return self.cnn[idx](x)
	
	def training_step(self, batch, batch_idx):
		optimizers = self.optimizers()
		for i in range(self.n_models):
			optimizers[i].zero_grad()
			loss = super().training_step(batch, i)
			self.manual_backward(loss)
			optimizers[i].step()
		
	def validation_step(self, batch, batch_idx):
		i = torch.randint(0, self.n_models, (1,)).item()
		return super().validation_step(batch, i)
	
	def test_step(self, batch, idx):
		x, y = batch['x'], batch['y']
		y_hat = torch.zeros_like(y)
		for i in range(self.n_models):
			y_hat += self.forward(x, i)
		y_hat /= self.n_models
		
		loss = self.loss(y_hat, y)
		self.log("test/rmse", self.denorm_rmse(loss))
		# self.log_images(x, y, y_hat, idx)

		self.test_predictions.append(y_hat)

		for channel in range(x.shape[1]):
			loss_ch = self.loss(x[:, channel:channel+1, :, :], y)
			self.log(f"test/rmse NWP {channel}", self.denorm_rmse(loss_ch))

		for metric in self.metrics:
			self.log(f"test/test_{metric.__name__}", metric(y_hat, y))

