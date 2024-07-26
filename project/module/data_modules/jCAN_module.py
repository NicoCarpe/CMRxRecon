import fastmri
import torch
from torch import nn
from argparse import ArgumentParser
from fastmri.data import transforms
from fastmri.pl_modules import MriModule
from models import ReconModel

class ReconMRIModule(MriModule):
    
    def __init__(
        self,
        lr: float = 0.0003,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        sparsity_tar: float = 0.25,
        shape: int = 320,
        coils: int = 24,
        num_heads : list = [3, 6, 12, 24],
        window_size : tuple = (8, 32, 64),
        depths : list = [2, 2, 18, 2],
        patch_size : tuple = (4,4,4),
        embed_dim : int = 96,
        mlp_ratio : float = 4.,
        mask: str = 'equispaced',
        use_amp: bool = False,
        num_recurrent: int = 25,
        sens_chans: int = 8,
        sens_steps: int = 4,
        lambda0: float = 10.0,
        lambda1: float = 10.0,
        lambda2: float = 1.0,
        lambda3: float = 100.0,
        GT: bool = True,
        n_SC: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Directly using the arguments
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.sparsity_tar = sparsity_tar
        self.shape = shape
        self.img_size = (shape, shape)
        self.coils = coils
        self.num_heads = num_heads
        self.window_size = window_size
        self.depths = depths
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.mask = mask
        self.use_amp = use_amp
        self.num_recurrent = num_recurrent
        self.sens_chans = sens_chans
        self.sens_steps = sens_steps
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.GT = GT
        self.n_SC = n_SC

        # Initialize the model with the provided configuration
        self.recon_model = ReconModel(coils=self.coils, img_size=self.img_size, num_heads=self.num_heads, window_size=self.window_size, 
                                      depths=self.depths, patch_size=self.patch_size, embed_dim=self.embed_dim, mlp_ratio=self.mlp_ratio, 
                                      qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, n_SC=self.n_SC,
                                      num_recurrent=self.num_recurrent, sens_chans=self.sens_chans, sens_steps=self.sens_steps, scale=0.1)

        self.loss = fastmri.SSIMLoss()

    def forward(self, masked_kspace, mask, num_low_frequencies):
        return self.recon_model(masked_kspace, mask, num_low_frequencies)

    def training_step(self, batch, batch_idx):
        masked_kspace, mask, num_low_frequencies, target, max_value = batch

        output = self(masked_kspace, mask, num_low_frequencies)
        target, output = transforms.center_crop_to_smallest(target, output)
        loss = self.loss(output, target)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        masked_kspace, mask, num_low_frequencies, target, max_value = batch

        output = self(masked_kspace, mask, num_low_frequencies)
        target, output = transforms.center_crop_to_smallest(target, output)
        val_loss = self.loss(output, target)

        self.log("val_loss", val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        masked_kspace, mask, num_low_frequencies, target, max_value = batch

        output = self(masked_kspace, mask, num_low_frequencies)
        target, output = transforms.center_crop_to_smallest(target, output)
        test_loss = self.loss(output, target)

        self.log("test_loss", test_loss)
        return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Define model parameters
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # param overwrites

        # training params (opt)
        parser.add_argument(
            "--lr", 
            default=0.0003, 
            type=float, 
            help="Adam learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40, 
            type=int, 
            help="Epoch at which to decrease step size"
        )
        parser.add_argument(
            "--lr_gamma", 
            default=0.1, 
            type=float, 
            help="Extent to which step size should be decreased"
        )
        parser.add_argument(
            "--weight_decay", 
            default=0.0, 
            type=float, 
            help="Strength of weight decay regularization"
        )
        parser.add_argument(
            "--use_checkpoint",
            action="store_true", 
            help="Use checkpoint (default: False)"
        )

        # network params
        parser.add_argument(
            '--sparsity_tar', 
            type=float, 
            default=0.25, 
            help='sparisity of masks for target modality'
        )
        parser.add_argument(
            '--shape', 
            type=int, 
            default=320, 
            help='image shape'
        )
        parser.add_argument(
            '--coils', 
            type=int, 
            default=24, 
            help='number of coils'
        )
        parser.add_argument(
            '--mask', 
            type=str, 
            default='equispaced', 
            choices=['mask', 'taylor', 'lowpass', 'equispaced', 'loupe','random'], 
            help='types of mask'
        )
        parser.add_argument(
            '--use_amp', 
            action='store_true', 
            help='use automatic mixed precision'
        )
        parser.add_argument(
            '--num_recurrent', 
            type=int, 
            default=25, 
            help='number of DCRBs'
        )
        parser.add_argument(
            '--sens_chans', 
            type=int, 
            default=8, 
            help='number of channels in sensitivity network'
        )
        parser.add_argument(
            '--sens_steps', 
            type=int, 
            default=4, 
            help='number of steps in initial sensitivity network'
        )
        parser.add_argument(
            '--lambda0', 
            type=float, 
            default=10, 
            help='weight of the kspace loss'
        )
        parser.add_argument(
            '--lambda1', 
            type=float, 
            default=10, 
            help='weight of the consistency loss in K-space'
        )
        parser.add_argument(
            '--lambda2', 
            type=float, 
            default=1, 
            help='weight of the SSIM loss'
        )
        parser.add_argument(
            '--lambda3', 
            type=float, 
            default=1e2, 
            help='weight of the TV Loss'
        )
        parser.add_argument(
            '--GT', 
            type=bool, 
            default=True, 
            help='if there is GT, default is True'
        )
        
        # Additional 3D Swin Transformer parameters
        parser.add_argument(
            '--num_heads', 
            type=int, 
            nargs='+', 
            default=[3, 6, 12, 24], 
            help='Number of attention heads for each transformer stage'
        )
        parser.add_argument(
            '--window_size', 
            type=int, 
            nargs=3, 
            default=(8, 32, 64), 
            help='Window size for each dimension (T, H, W)'
        )
        parser.add_argument(
            '--depths', 
            type=int, 
            nargs='+', 
            default=[2, 2, 18, 2], 
            help='Depths of each Swin Transformer stage'
        )
        parser.add_argument(
            '--patch_size', 
            type=int, 
            nargs=3, 
            default=(4, 4, 4), 
            help='Patch size for each dimension (T, H, W)'
        )
        parser.add_argument(
            '--embed_dim', 
            type=int, 
            default=96, 
            help='Embedding dimension'
        )
        parser.add_argument(
            '--mlp_ratio', 
            type=float, 
            default=4., 
            help='Ratio of mlp hidden dim to embedding dim'
        )
        parser.add_argument(
            '--min_windows', 
            type=int, 
            default=4, 
            help='Minimum number of windows needed for regularization of data'
        )
        return parser
