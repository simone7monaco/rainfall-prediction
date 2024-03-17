import argparse
from pathlib import Path

import wandb
from torch import cuda
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from models import SegmentationModel


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--case_study", type=str, default="24h_10mmMAX_OI", choices=['24h_10mmMAX_OI', '24h_10mmMAX_radar'])
	parser.add_argument("--network_model", type=str, default="unet_2")
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--split_idx", type=str, default="701515")
	parser.add_argument("--n_split", type=int, default=9)
	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--epochs", type=int, default=100)
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()
	return args
	

def main(args):
	pl.seed_everything(args.seed)
	scratch_path = Path("/home/students/s265780/data")
	# scratch_path = Path("/home/monaco/MultimodelPreci")

	input_path = scratch_path / args.case_study
	output_path = Path('reports')
	output_path /= f'{args.network_model}'
	
	args.input_path = input_path
	args.output_path = output_path
    
	# logger = CSVLogger(output_path, name=args.network_model)
    
	# initialise the wandb logger and name your wandb project
	if args.epochs > 1:
		wandb_logger = WandbLogger(project='rainfall_prediction')   
		# add your batch size to the wandb config
		wandb_logger.experiment.config["batch_size"] = 32
	else:
		wandb_logger = CSVLogger(output_path, name=args.network_model)


	early_stop = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10, verbose=False, mode="min")
	model_checkpoint = ModelCheckpoint(output_path / args.network_model, monitor='val_loss', mode='min', filename='{epoch}-{val_rmse:.2f}')
	lr_monitor = LearningRateMonitor(logging_interval='step')
    
	model = SegmentationModel(**args.__dict__)
	trainer = pl.Trainer(
        accelerator='gpu' if cuda.is_available() else 'cpu',
        max_epochs=args.epochs,
        callbacks=[model_checkpoint, lr_monitor],
		log_every_n_steps=1,
        logger=wandb_logger, # default is TensorBoard
    )
	trainer.fit(model)

	print(f"\nLoading best model ({model_checkpoint.best_model_path})")
	model = SegmentationModel.load_from_checkpoint(model_checkpoint.best_model_path)
	trainer.test(model)
	# trainer.test(model, model.train_dataloader())
	# trainer.test(model, model.val_dataloader())


if __name__ == '__main__':
	args = get_args()
	main(args)