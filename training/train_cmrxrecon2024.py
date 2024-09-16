import os
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.profilers import PyTorchProfiler
import yaml
from pathlib import Path
from data_utils.transforms import MRIDataTransform
from data_modules.cmrxrecon2024_data_module import CmrxReconDataModule
from data_modules.model_module import ReconMRIModule
import torch  # Import torch
from torch.profiler import schedule, tensorboard_trace_handler, ProfilerActivity


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
    model = ReconMRIModule(**vars(args))  # Unpack args as kwargs

    # ------------
    # logger
    # ------------
    output_dir = Path(args.output_dir) / "logs"
    logger = TensorBoardLogger(
        save_dir=output_dir, 
        name="ReconMRI",
        log_graph=True
    )

    # ------------
    # profiler
    # ------------
    
    profiler = PyTorchProfiler(
        dirpath=output_dir / "profile",
        filename="profiler_trace",
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],  # Capture both CPU and GPU
        schedule=schedule(wait=1, warmup=1, active=5, repeat=2),
        on_trace_ready=tensorboard_trace_handler(output_dir / "profile"),
        with_stack=True,
        profile_memory=True,  # Capture memory usage
        record_shapes=True  # Record tensor shapes
)

    # ------------
    # checkpoint callback
    # ------------
    args.callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            save_top_k=-1,  # Save only the best model
            verbose=True,
            monitor="validation_loss",
            mode="min"
        )
    ]

    # Logic for handling checkpoint resumption
    resume_checkpoint_path = None
    if args.resume_from_checkpoint:
        ckpt_list = sorted(args.checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            resume_checkpoint_path = str(ckpt_list[-1])
            print(f"Resuming from the latest checkpoint: {resume_checkpoint_path}")
        else:
            print(f"Warning: No checkpoints found in {args.checkpoint_dir}. Starting training from scratch.")
    else:
        print("Starting training from scratch.")

    # ------------
    # trainer setup with DDPStrategy
    # ------------
    trainer = pl.Trainer(
        logger=logger,
        callbacks=args.callbacks,
        fast_dev_run=False,
        strategy=DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True),  # Use DDP with gradient bucket optimization
        devices=args.num_gpus,
        accelerator="gpu",
        max_epochs=args.max_epochs,
        gradient_clip_val=args.gradient_clip_val,  # Optional gradient clipping
        precision="16-mixed" if args.use_amp else 32,  # AMP if enabled, otherwise 32-bit precision
        profiler=profiler,  # Add the profiler
    )

    # ------------
    # run
    # ------------
    trainer.fit(model, datamodule=data_module, ckpt_path=resume_checkpoint_path)

    # Print CPU times
    print(profiler.key_averages().table(sort_by="cpu_time_total"), flush=True)

    # Print GPU times
    print(profiler.key_averages().table(sort_by="cuda_time_total"), flush=True)

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
