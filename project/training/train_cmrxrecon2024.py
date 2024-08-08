import os
import sys
import pathlib
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(pathlib.Path(__file__).parent.absolute())))

from data_utils.transforms import MRIDataTransform
from data_modules.cmrxrecon2024_data_module import CmrxReconDataModule
from data_modules.model_module import ReconMRIModule

def cli_main(args):
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    train_transform = MRIDataTransform(mask_func=None)
    val_transform = MRIDataTransform(mask_func=None)

    # pl data module - this handles data loaders
    data_module = CmrxReconDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        combine_train_val=args.combine_train_val,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=(args.strategy in (
            "ddp_find_unused_parameters_false", "ddp", "ddp_cpu")),
    )

    # ------------
    # model
    # ------------
    model = ReconMRIModule(
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
        coils=args.coils,
        num_heads=args.num_heads,
        window_size=args.window_size,
        depths=args.depths,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        mlp_ratio=args.mlp_ratio,
        use_amp=args.use_amp,
        num_recurrent=args.num_recurrent,
        sens_chans=args.sens_chans,
        sens_steps=args.sens_steps,
        lambda0=args.lambda0,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        lambda3=args.lambda3,
        GT=args.GT,
        n_SC=args.n_SC
    )

    # ------------
    # logger
    # ------------
    logger = TensorBoardLogger("logs/", name="ReconMRI")

    # ------------
    # trainer
    # ------------
    trainer = pl.Trainer(
        logger=logger,
        strategy="ddp",
        devices=args.num_gpus,
        accelerator="gpu",
        max_epochs=args.max_epochs,
        gradient_clip_val=args.gradient_clip_val,
        precision=16 if args.use_amp else 32,
    )

    # ------------
    # run
    # ------------
    trainer.fit(model, datamodule=data_module)

def build_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the YAML config file')
    args = parser.parse_args()

    # Load config from YAML file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    for key, value in config.items():
        setattr(args, key, value)

    return args

def run_cli():
    args = build_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    cli_main(args)

if __name__ == "__main__":
    run_cli()
