import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import argparse
from pathlib import Path

from torch import cuda
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import yaml
import wandb
from utils import str2bool


def get_args(args=None):
	parser = argparse.ArgumentParser()
	parser.add_argument("--case_study", "-c", type=str, default="24h_10mmMAX_OI", choices=['24h_10mmMAX_OI', '24h_10mmMAX_radar'])
	parser.add_argument("--network_model", "-m", type=str, default="unet", choices=['unet', 'sde_unet', 'ensemble_unet'])
	parser.add_argument("--batch_size", type=int, default=32)
	# parser.add_argument("--split_idx", type=str, default="701515")
	parser.add_argument("--n_split", type=int, default=8)
	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--epochs", "-e", type=int, default=300)
	parser.add_argument("--mcdropout", type=float, default=0)
	parser.add_argument("--load_checkpoint", type=Path, default=None)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--wb", type=str2bool, nargs='?', const=True, default=False)
	parser.add_argument("--eval_proba", type=str2bool, nargs='?', const=True, default=False)
	args = parser.parse_args(args)
	return args
	

def main(args):
	pl.seed_everything(args.seed)

	scratch_path = Path("/media/monaco/DATA1/case_study")
	conf_dev = Path("config_dev.yaml")
	if conf_dev.exists():
		with open(conf_dev, 'r') as f:
			conf_dev = yaml.safe_load(f)
		scratch_path = Path(conf_dev['scratch_path'])

	input_path = scratch_path / args.case_study
	output_path = Path('lightning_logs')
	output_path /= f'{args.network_model}'
	
	args.input_path = input_path
    
	# logger = CSVLogger(output_path, name=args.network_model)

	if not args.load_checkpoint:
		early_stop = EarlyStopping(monitor="val/loss", min_delta=0.00, patience=10, verbose=False, mode="min")
		model_checkpoint = ModelCheckpoint(output_path / f"split_{args.n_split}", monitor='val/loss', mode='min', filename='{epoch}-{val_rmse:.2f}')
		
		if args.network_model == 'unet':
			from base_segmodel import _SegmentationModel as SegmentationModel
		elif args.network_model == 'sde_unet':
			from sde_segmodel import SegmentationModel
		elif args.network_model == 'ensemble_unet':
			from ensemble_segmodel import SegmentationModel
		else:
			raise ValueError(f"Unknown network model {args.network_model}")

		if args.wb:
			logger = pl.loggers.WandbLogger(project='rainfall')
			logger.log_hyperparams(args.__dict__)
		else:
			logger = None
		model = SegmentationModel(**args.__dict__)
		trainer = pl.Trainer(
			accelerator='gpu' if cuda.is_available() else 'cpu',
			max_epochs=args.epochs,
			callbacks=[model_checkpoint],
			log_every_n_steps=1,
			num_sanity_val_steps=0,
			logger=logger,
		)
		trainer.fit(model)

		print(f"\nLoading best model ({model_checkpoint.best_model_path})")
		model = SegmentationModel.load_from_checkpoint(model_checkpoint.best_model_path)
	else:
		trainer = pl.Trainer(accelerator='gpu' if cuda.is_available() else 'cpu')
		print(f"\n⬆️  Loading checkpoint {args.load_checkpoint}")
		model = SegmentationModel.load_from_checkpoint(args.load_checkpoint)
		
	
	trainer.test(model)
	if args.eval_proba:
		model.eval_proba(save_dir=Path('proba'))
		


if __name__ == '__main__':
	args = get_args()
	if args.mcdropout and args.network_model == 'sde_unet':
		print("MC Dropout is not implemented for SDE-UNet")
		exit()
	main(args)