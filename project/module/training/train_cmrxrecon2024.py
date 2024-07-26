import os
import sys
import pathlib
from argparse import ArgumentParser
import pytorch_lightning as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(pathlib.Path(__file__).parent.absolute())))

from data_utils.transforms import MRIDataTransform
from data_modules.cmrxrecon2024_data_module import CmrxReconDataModule
from data_modules.jCAN_module import ReconMRIModule
from data_utils.subsample import create_mask_for_mask_type

def cli_main(args):
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------

    # this creates a k-space mask for transforming input data
    mask = create_mask_for_mask_type(
        args.mask_type, None, args.accelerations, args.center_numbers
    )

    # use equispaced_fixed masks for train transform, fixed masks for val transform
    train_transform = MRIDataTransform(mask_func=mask, use_seed=False)
    val_transform = MRIDataTransform(mask_func=mask)
    test_transform = MRIDataTransform()

    # ptl data module - this handles data loaders
    data_module = CmrxReconDataModule(
        data_path=args.data_path,
        h5py_folder=args.h5py_folder,
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        combine_train_val=args.combine_train_val,
        test_split=args.test_split,
        test_path=args.test_path,
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
        sparsity_tar=args.sparsity_tar,
        shape=args.shape,
        coils=args.coils,
        num_heads=args.num_heads,
        window_size=args.window_size,
        depths=args.depths,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        mlp_ratio=args.mlp_ratio,
        mask=args.mask,
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
    # trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)

    # ------------
    # run
    # ------------
    if args.mode == "train":
        trainer.fit(model, datamodule=data_module)
    elif args.mode == "test":
        trainer.test(model, datamodule=data_module)
    else:
        raise ValueError(f"unrecognized mode {args.mode}")

def build_args():
    parser = ArgumentParser()

    # client arguments
    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "test"),
        type=str,
        help="Operation mode",
    )
    parser.add_argument(
        "--num_gpus",
        default=1,
        type=int,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "--exp_name",
        default="promptmr_train",
        type=str,
        help="experiment name",
    )

    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced_fraction", 'equispaced_fixed'),
        default="random",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--center_numbers",
        nargs="+",
        default=[16],
        type=int,
        help="Number of center lines to use in mask",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[4, 8, 12, 16, 20, 24],
        type=int,
        help="Acceleration rates to use for masks",
    )

    # Add data module specific arguments
    parser = CmrxReconDataModule.add_data_specific_args(parser)
    
    # Add model specific arguments
    parser = ReconMRIModule.add_model_specific_args(parser)
    
    # Add trainer specific arguments
    parser = pl.Trainer.add_argparse_args(parser)

    # Set default arguments
    data_path = pathlib.Path('.')
    default_root_dir = data_path / "experiments"
    parser.set_defaults(
        replace_sampler_ddp=False,
        strategy="ddp_find_unused_parameters_false",
        seed=42,
        deterministic=False,
        max_epochs=25,
        gradient_clip_val=0.01
    )

    args = parser.parse_args()
    args.gpus = args.num_gpus  # override pl.Trainer gpus arg
    acc_folder = "acc_" + "_".join(map(str, args.accelerations))
    args.default_root_dir = default_root_dir / args.exp_name / acc_folder

    # Configure checkpointing in checkpoint_dir
    checkpoint_dir = args.default_root_dir / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=args.default_root_dir / "checkpoints",
            save_top_k=True,
            verbose=True,
            monitor="val_loss",  # Use "val_loss" instead of "validation_loss" to match the actual log key
            mode="min",
        )
    ]

    # Set default checkpoint if one exists in our checkpoint directory
    if args.resume_from_checkpoint is None:
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            args.resume_from_checkpoint = str(ckpt_list[-1])

    return args

def run_cli():
    args = build_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    cli_main(args)

if __name__ == "__main__":
    run_cli()
