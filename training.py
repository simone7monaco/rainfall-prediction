import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
from pathlib import Path

import wandb
import torch
from torch import cuda
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import yaml

from temperature_scaling import ModelWithTemperature


from models import SegmentationModel


def get_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_study", "-c", type=str, default="RYDL", choices=["24h_10mmMAX_OI", "24h_10mmMAX_radar", "RYDL"])
    parser.add_argument("--network_model", "-m", type=str, default="unet")
    parser.add_argument("--batch_size", type=int, default=32)
    # parser.add_argument("--split_idx", type=str, default="701515")
    parser.add_argument("--n_split", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--load_checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--forward_passes", type=int, default=1)
    parser.add_argument("--code_version", type=int, default=5)
    parser.add_argument("--fine_tune", type=int, default=1)
    parser.add_argument("--n_thresh", type=int, default=7)
    parser.add_argument("--indx_thresh", type=int, default=0)
    parser.add_argument("--epochs_fn", "-f", type=int, default=50)
    parser.add_argument("--finetune_type", type=str, default="bin", choices=["mine", "bin", "kde"])
    args = parser.parse_args(args)
    return args


def main(args):
    pl.seed_everything(args.seed)
    scratch_path = Path("/home/students/s265780/data")
    conf_dev = Path("config_dev.yaml")
    if conf_dev.exists():
        with open(conf_dev, "r") as f:
            conf_dev = yaml.safe_load(f)
        scratch_path = Path(conf_dev["scratch_path"])

    input_path = scratch_path / args.case_study
    output_path = Path("lightning_logs")
    output_path /= f"{args.network_model}"

    args.input_path = input_path
    fine_tune = 0
    if args.fine_tune == 1:
        fine_tune = 1
        args.fine_tune = 0

    logger = WandbLogger(project="rainfall_prediction")
    # add your batch size to the wandb config
    logger.experiment.config["batch_size"] = 32

    if  args.epochs>0:
        if not args.load_checkpoint:
            early_stop = EarlyStopping(monitor="val/loss", min_delta=0.00, patience=15, verbose=False, mode="min")
            model_checkpoint = ModelCheckpoint(
                output_path / f"split_{args.n_split}",
                monitor="val/loss",
                mode="min",
                filename="{epoch}-{val/loss:.2f}",
            )

            model = SegmentationModel(**args.__dict__)
            trainer = pl.Trainer(
                accelerator="gpu" if cuda.is_available() else "cpu",
                max_epochs=args.epochs,
                callbacks=[model_checkpoint, early_stop],
                log_every_n_steps=1,
                logger=logger,  # default is TensorBoard
            )
            trainer.fit(model)

            print(f"\nLoading best model ({model_checkpoint.best_model_path})")
            model = SegmentationModel.load_from_checkpoint(
                model_checkpoint.best_model_path,
                fine_tune=fine_tune,
                finetune_type=args.finetune_type,
            )
            
            
            
            if fine_tune == 0:
                temperature=0 #set here
                if temperature==1:
                    model_temperature = ModelWithTemperature(model, args.seed, args.n_split, args.input_path, args.case_study, args.n_thresh)
                    temp = model_temperature.set_temperature()
                
                trainer.test(model)
        else:
            trainer = pl.Trainer(accelerator="gpu" if cuda.is_available() else "cpu")
            print(f"\n⬆️  Loading checkpoint {args.load_checkpoint}")
            model = SegmentationModel.load_from_checkpoint(
                args.load_checkpoint, fine_tune=fine_tune, finetune_type=args.finetune_type
            )
    else:
        args.fine_tune = 1
        model = SegmentationModel(**args.__dict__)
    

    if fine_tune == 1:
        model_checkpoint = ModelCheckpoint(
            output_path / f"split_{args.n_split}",
            monitor="val/loss",
            mode="min",
            filename="{epoch}-{val/loss:.4f}",
        )
        trainer = pl.Trainer(
            accelerator="gpu" if cuda.is_available() else "cpu",
            max_epochs=args.epochs_fn,
            callbacks=[model_checkpoint],
            log_every_n_steps=1,
            logger=logger,  # default is TensorBoard
        )
        trainer.fit(model)

        print(f"\nLoading best model ({model_checkpoint.best_model_path})")
        model = SegmentationModel.load_from_checkpoint(
            model_checkpoint.best_model_path,
            fine_tune=fine_tune,
            finetune_type=args.finetune_type,
        )

        trainer.test(model)

if __name__ == "__main__":
    args = get_args()
    main(args)
