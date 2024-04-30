import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
from pathlib import Path

import wandb
from torch import cuda
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import yaml


from models import SegmentationModel


def get_args(args=None):
	parser = argparse.ArgumentParser()
	parser.add_argument("--case_study", "-c", type=str, default="24h_10mmMAX_OI", choices=['24h_10mmMAX_OI', '24h_10mmMAX_radar'])
	parser.add_argument("--network_model", "-m", type=str, default="unet")
	parser.add_argument("--batch_size", type=int, default=32)
	# parser.add_argument("--split_idx", type=str, default="701515")
	parser.add_argument("--n_split", type=int, default=8)
	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--epochs", "-e", type=int, default=150)
	parser.add_argument("--mcdropout", type=float, default=0)
	parser.add_argument("--load_checkpoint", type=Path, default=None)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--forward_passes", type=int, default=1)
	parser.add_argument("--code_version", type=int, default=3)
	args = parser.parse_args(args)
	return args
	

def main(args):
	pl.seed_everything(args.seed)
	scratch_path = Path("/home/students/s265780/data")
	conf_dev = Path("config_dev.yaml")
	if conf_dev.exists():
		with open(conf_dev, 'r') as f:
			conf_dev = yaml.safe_load(f)
		scratch_path = Path(conf_dev['scratch_path'])

	input_path = scratch_path / args.case_study
	output_path = Path('lightning_logs')
	output_path /= f'{args.network_model}'
	
	args.input_path = input_path
    
	if args.epochs > 1 and not args.load_checkpoint:
		logger = WandbLogger(project='rainfall_prediction')   
		# add your batch size to the wandb config
		logger.experiment.config["batch_size"] = 32
	else:
		logger = WandbLogger(project='rainfall_prediction_trash')

	if not args.load_checkpoint:
		early_stop = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10, verbose=False, mode="min")
		model_checkpoint = ModelCheckpoint(output_path / f"split_{args.n_split}", monitor='val_loss', mode='min', filename='{epoch}-{val_rmse:.2f}')
		
		model = SegmentationModel(**args.__dict__)
		trainer = pl.Trainer(
			accelerator='gpu' if cuda.is_available() else 'cpu',
			max_epochs=args.epochs,
			callbacks=[model_checkpoint],
			log_every_n_steps=1,
			logger=logger # default is TensorBoard
		)
		trainer.fit(model)

		print(f"\nLoading best model ({model_checkpoint.best_model_path})")
		model = SegmentationModel.load_from_checkpoint(model_checkpoint.best_model_path)
	else:
		trainer = pl.Trainer(accelerator='gpu' if cuda.is_available() else 'cpu')
		print(f"\n⬆️  Loading checkpoint {args.load_checkpoint}")
		model = SegmentationModel.load_from_checkpoint(args.load_checkpoint)
		

	if model.hparams.mcdropout:
		model.get_monte_carlo_predictions(forward_passes=args.forward_passes, save_dir=Path('reports'))
		model.eval_proba(save_dir=Path('proba'), forward_passes=args.forward_passes)
	else:
		trainer.test(model)


if __name__ == '__main__':
	args = get_args()
	main(args)