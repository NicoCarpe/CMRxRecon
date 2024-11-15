import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from typing import Union, List
import numpy as np
from timm.models.layers import trunc_normal_
from model.triton.layer_norm import LayerNorm


def layer_norm(x):
    # LayerNorm parameters
    normalized_shape = x.shape[-1]
    weight = torch.ones(normalized_shape, device=x.device, dtype=x.dtype)
    bias = torch.zeros(normalized_shape, device=x.device, dtype=x.dtype)
    eps = 1e-5

    # Using Triton-based LayerNorm
    layer_norm = LayerNorm.apply
    y_triton = layer_norm(x, normalized_shape, weight, bias, eps)

    return y_triton


class Residual3D(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm3D(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        #self.norm = layer_norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward3D(nn.Module):
    def __init__(self, dim, hidden_dim, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.net(x)
        x = self.drop(x)
        return x


def create_mask3D(window_size: Union[int, List[int]], displacement: Union[int, List[int]],
                  t_shift: bool, x_shift: bool, y_shift: bool):
    assert type(window_size) is int or len(window_size) == 3, f'window_size must be 1 or 3 dimensions'
    if type(window_size) is int:
        window_size = np.array([window_size] * 3)

    assert type(displacement) is int or len(displacement) == 3, f'displacement must be 1 or 3 dimensions'
    if type(displacement) is int:
        displacement = np.array([displacement] * 3)

    assert len(window_size) == len(displacement)
    for i in range(len(window_size)):
        assert 0 < displacement[i] < window_size[i], \
            f'Invalid displacement on axis {i}. Dimensions include T(i=0), X(i=1), and Y(i=2)'

    mask = torch.zeros(window_size[0] * window_size[1] * window_size[2],
                       window_size[0] * window_size[1] * window_size[2])  # (wt*wx*wy, wt*wx*wy)
    mask = rearrange(mask, '(t1 x1 y1) (t2 x2 y2) -> t1 x1 y1 t2 x2 y2',
                     x1=window_size[1], y1=window_size[2], x2=window_size[1], y2=window_size[2])

    t_dist, x_dist, y_dist = displacement

    if t_shift:
        mask[-t_dist:, :, :, :-t_dist, :, :] = float('-inf')
        mask[:-t_dist, :, :, -t_dist:, :, :] = float('-inf')

    if x_shift:
        mask[:, -x_dist:, :, :, :-x_dist, :] = float('-inf')
        mask[:, :-x_dist, :, :, -x_dist:, :] = float('-inf')

    if y_shift:
        mask[:, :, -y_dist:, :, :, :-y_dist] = float('-inf')
        mask[:, :, :-y_dist, :, :, -y_dist:] = float('-inf')

    mask = rearrange(mask, 't1 x1 y1 t2 x2 y2 -> (t1 x1 y1) (t2 x2 y2)')
    return mask


class WindowAttention3D(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size: Union[int, List[int]], relative_pos_embedding=True):
        super().__init__()
        inner_dim = head_dim * heads
        self.heads = heads
        self.window_size = np.array(window_size) if isinstance(window_size, list) else np.array([window_size] * 3)
        self.scale = head_dim ** -0.5

        self.shifted = shifted
        if self.shifted:
            self.shift_size = self.window_size // 2
        else:
            self.shift_size = np.array([0, 0, 0])

        # Masking to handle attention for shifted windows
        if self.shifted:
            self.t_mask = nn.Parameter(create_mask3D(self.window_size, self.shift_size, t_shift=True, x_shift=False, y_shift=False), requires_grad=False)
            self.x_mask = nn.Parameter(create_mask3D(self.window_size, self.shift_size, t_shift=False, x_shift=True, y_shift=False), requires_grad=False)
            self.y_mask = nn.Parameter(create_mask3D(self.window_size, self.shift_size, t_shift=False, x_shift=False, y_shift=True), requires_grad=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        if self.shifted:
            x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))

        b, n_t, n_x, n_y, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        nw_t = n_t // self.window_size[0]
        nw_x = n_x // self.window_size[1]
        nw_y = n_y // self.window_size[2]

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_t w_t) (nw_x w_x) (nw_y w_y) (h d) -> b h (nw_t nw_x nw_y) (w_t w_x w_y) d',
                                h=h, w_t=self.window_size[0], w_x=self.window_size[1], w_y=self.window_size[2]), qkv)

        # similarity scores between the vectors from q and k for each combination of the indices i and j
        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        # Apply window-specific masking for shifted windows
        if self.shifted:
            dots = rearrange(dots, 'b h (n_t n_x n_y) i j -> b h n_x n_y n_t i j', n_x=nw_x, n_y=nw_y)
            dots[:, :, :, :, -1] += self.t_mask
            dots = rearrange(dots, 'b h n_x n_y n_t i j -> b h n_t n_y n_x i j')
            dots[:, :, :, :, -1] += self.x_mask
            dots = rearrange(dots, 'b h n_t n_y n_x i j -> b h n_t n_x n_y i j')
            dots[:, :, :, :, -1] += self.y_mask
            dots = rearrange(dots, 'b h n_t n_x n_y i j -> b h (n_t n_x n_y) i j')

        attn = self.softmax(dots)
        
        # performs a weighted sum of the values
        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)

        out = rearrange(out, 'b h (nw_t nw_x nw_y) (w_t w_x w_y) d -> b (nw_t w_t) (nw_x w_x) (nw_y w_y) (h d)',
                        h=h, w_t=self.window_size[0], w_x=self.window_size[1], w_y=self.window_size[2],
                        nw_t=nw_t, nw_x=nw_x, nw_y=nw_y)

        out = self.to_out(out)

        if self.shifted:
            out = torch.roll(out, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))

        return out



class SwinBlock3D(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding=True, dropout=0.0):
        super().__init__()

        self.shifted = shifted  # Whether to use the shifted window mechanism
        self.window_size = window_size  # Window size

        # Residual attention block with or without shift
        self.attention_block = Residual3D(
            PreNorm3D(dim, WindowAttention3D(
                dim=dim,
                heads=heads,
                head_dim=head_dim,
                shifted=shifted,
                window_size=window_size,
                relative_pos_embedding=relative_pos_embedding
            ))
        )

        # MLP block after attention with residual
        self.mlp_block = Residual3D(
            PreNorm3D(dim, FeedForward3D(dim=dim, hidden_dim=mlp_dim, dropout=dropout))
        )

    def forward(self, x):
        # Apply attention block (with or without shift depending on the value of `shifted`)
        x = self.attention_block(x)

        # Apply the feed-forward MLP block
        x = self.mlp_block(x)

        return x


class Norm(nn.Module):
    def __init__(self, dim, channel_first: bool = True):
        super(Norm, self).__init__()
        if channel_first:
            self.net = nn.Sequential(
                Rearrange('b c t x y -> b t x y c'),
                nn.LayerNorm(dim),
                # layer_norm(dim)
                Rearrange('b t x y c -> b c t x y')
            )

            # self.net = nn.InstanceNorm3d(dim, eps=1e-5, momentum=0.1, affine=False)
        else:
            self.net = nn.LayerNorm(dim)
            # self.net = layer_norm(dim)

    def forward(self, x):
        x = self.net(x)
        return x


class PatchMerging3D(nn.Module):
    def __init__(self, in_dim, out_dim, downscaling_factor):
        super().__init__()
        dsf = downscaling_factor
        self.net = nn.Sequential(
            Rearrange('b c (n_t dsf_t) (n_x dsf_x) (n_y dsf_y) -> b n_t n_x n_y (dsf_t dsf_x dsf_y c)',
                dsf_t=dsf, dsf_x=dsf, dsf_y=dsf),
            nn.Linear(in_dim * (dsf ** 3), out_dim),
            Norm(out_dim, channel_first=False),
            Rearrange('b t x y c -> b c t x y')
        )

    def forward(self, x):
        # x: B, C, T, X, Y
        x = self.net(x)
        return x  # B, T //down_scaling, X // down_scaling, Y // down_scaling, out_dim



class PatchExpanding3D(nn.Module):
    def __init__(self, in_dim, out_dim, up_scaling_factor):
        super(PatchExpanding3D, self).__init__()

        self.usf = up_scaling_factor
        hidden_dim = (up_scaling_factor ** 3) * out_dim
        self.net = nn.Sequential(
            Rearrange('b c t x y -> b t x y c'),
            nn.Linear(in_dim, hidden_dim),
            Rearrange('b t_s x_s y_s (fac1 fac2 fac3 c) -> b c (t_s fac1) (x_s fac2) (y_s fac3)',
                      fac1=self.usf, fac2=self.usf, fac3=self.usf),
            Norm(out_dim),
        )

    def forward(self, x):
        '''X: B,C,T,X,Y'''
        x = self.net(x)
        return x


class FinalExpand3D(nn.Module):  # Final voxel classification
    def __init__(self, in_dim, out_dim, up_scaling_factor):
        super(FinalExpand3D, self).__init__()
        # T, X, Y, out_dims -> T, X, Y, (up_scaling ** 3) * out_dims
        self.usf = up_scaling_factor
        hidden_dim = (up_scaling_factor ** 3) * out_dim
        self.net = nn.Sequential(
            Rearrange('b c t x y -> b t x y c'),
            nn.Linear(in_dim, hidden_dim),
            Rearrange('b t_s x_s y_s (fac1 fac2 fac3 c) -> b c (t_s fac1) (x_s fac2) (y_s fac3)',
                      fac1=self.usf, fac2=self.usf, fac3=self.usf),
            Norm(out_dim),
            nn.PReLU()
        )

    def forward(self, x):
        '''X: B,C,T,X,Y'''
        x = self.net(x)
        return x



class Encoder(nn.Module):
    def __init__(self, in_dims, hidden_dimension, layers, downscaling_factor, num_heads, head_dim,
                 window_size: Union[int, List[int]], relative_pos_embedding: bool = True, dropout: float = 0.0):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchMerging3D(in_dim=in_dims, out_dim=hidden_dimension,
                                              downscaling_factor=downscaling_factor)

        self.re1 = Rearrange('b c t x y -> b t x y c')
        self.swin_layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.swin_layers.append(nn.ModuleList([
                SwinBlock3D(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                            shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                            dropout=dropout),
                SwinBlock3D(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                            shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                            dropout=dropout),
            ]))
        self.re2 = Rearrange('b t x y c -> b c t x y')

    def forward(self, x):
        x = self.patch_partition(x)

        x = self.re1(x)
        for regular_block, shifted_block in self.swin_layers:  # swin_layers块学习长距离依赖
            x = regular_block(x)
            x = shifted_block(x)
        x = self.re2(x)

        return x


class Decoder(nn.Module):
    def __init__(self, in_dims, out_dims, layers, up_scaling_factor, num_heads, head_dim,
                 window_size: Union[int, List[int]], relative_pos_embedding, dropout: float = 0.0):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_expand = PatchExpanding3D(in_dim=in_dims, out_dim=out_dims,
                                             up_scaling_factor=up_scaling_factor)

        self.re1 = Rearrange('b c t x y -> b t x y c')
        self.swin_layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.swin_layers.append(nn.ModuleList([
                SwinBlock3D(dim=out_dims, heads=num_heads, head_dim=head_dim, mlp_dim=out_dims * 4,
                            shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                            dropout=dropout),
                SwinBlock3D(dim=out_dims, heads=num_heads, head_dim=head_dim, mlp_dim=out_dims * 4,
                            shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                            dropout=dropout),
            ]))
        self.re2 = Rearrange('b t x y c -> b c t x y')

    def forward(self, x):
        x = self.patch_expand(x)

        x = self.re1(x)
        for regular_block, shifted_block in self.swin_layers:
            x = regular_block(x)
            x = shifted_block(x)
        x = self.re2(x)

        return x


class Converge(nn.Module):
    def __init__(self, dim: int):
        '''
        stack:融合方式以堆叠+线性变换实现
        add 跳跃连接通过直接相加的方式实现
        '''
        super(Converge, self).__init__()
        self.norm = Norm(dim=dim)

    def forward(self, x, enc_x):
        '''
         x: B,C,T,X,Y
        enc_x:B,C,T,X,Y
        '''
        assert x.shape == enc_x.shape
        x = x + enc_x
        x = self.norm(x)
        return x


class SwinUnet3D(nn.Module):
    def __init__(self, *, hidden_dim, layers, heads, in_channel=2, num_classes=2, head_dim=32,
                 window_size: Union[int, List[int]] = 7, downscaling_factors=(2, 2, 2, 2),
                 relative_pos_embedding=True, dropout: float = 0.0, skip_style='stack',
                 stl_channels: int = 32):  # second_to_last_channels
        super().__init__()

        self.dsf = downscaling_factors
        self.window_size = window_size

        self.enc12 = Encoder(in_dims=in_channel, hidden_dimension=hidden_dim, layers=layers[0],
                             downscaling_factor=downscaling_factors[0], num_heads=heads[0],
                             head_dim=head_dim, window_size=window_size, dropout=dropout,
                             relative_pos_embedding=relative_pos_embedding)
        self.enc3 = Encoder(in_dims=hidden_dim, hidden_dimension=hidden_dim * 2,
                            layers=layers[1],
                            downscaling_factor=downscaling_factors[1], num_heads=heads[1],
                            head_dim=head_dim, window_size=window_size, dropout=dropout,
                            relative_pos_embedding=relative_pos_embedding)
        self.enc4 = Encoder(in_dims=hidden_dim * 2, hidden_dimension=hidden_dim * 4,
                            layers=layers[2],
                            downscaling_factor=downscaling_factors[2], num_heads=heads[2],
                            head_dim=head_dim, window_size=window_size, dropout=dropout,
                            relative_pos_embedding=relative_pos_embedding)
        self.enc5 = Encoder(in_dims=hidden_dim * 4, hidden_dimension=hidden_dim * 8,
                            layers=layers[3],
                            downscaling_factor=downscaling_factors[3], num_heads=heads[3],
                            head_dim=head_dim, window_size=window_size, dropout=dropout,
                            relative_pos_embedding=relative_pos_embedding)

        self.dec4 = Decoder(in_dims=hidden_dim * 8, out_dims=hidden_dim * 4,
                            layers=layers[2],
                            up_scaling_factor=downscaling_factors[3], num_heads=heads[2],
                            head_dim=head_dim, window_size=window_size, dropout=dropout,
                            relative_pos_embedding=relative_pos_embedding)

        self.dec3 = Decoder(in_dims=hidden_dim * 4, out_dims=hidden_dim * 2,
                            layers=layers[1],
                            up_scaling_factor=downscaling_factors[2], num_heads=heads[1],
                            head_dim=head_dim, window_size=window_size, dropout=dropout,
                            relative_pos_embedding=relative_pos_embedding)

        self.dec12 = Decoder(in_dims=hidden_dim * 2, out_dims=hidden_dim,
                             layers=layers[0],
                             up_scaling_factor=downscaling_factors[1], num_heads=heads[0],
                             head_dim=head_dim, window_size=window_size, dropout=dropout,
                             relative_pos_embedding=relative_pos_embedding)

        self.converge4 = Converge(hidden_dim * 4)
        self.converge3 = Converge(hidden_dim * 2)
        self.converge12 = Converge(hidden_dim)

        self.final = FinalExpand3D(in_dim=hidden_dim, out_dim=stl_channels,
                                   up_scaling_factor=downscaling_factors[0])
        self.out = nn.Sequential(
            # nn.Linear(stl_channels, num_classes),
            # Rearrange('b t x y c -> b c t x y'),
            nn.Conv3d(stl_channels, num_classes, kernel_size=1)
        )
        # 参数初始化
        self.init_weight()

    def forward(self, img):
        window_size = self.window_size
        assert type(window_size) is int or len(window_size) == 3, f'window_size must be 1 or 3 dimensions'
        if type(window_size) is int:
            window_size = np.array([window_size, window_size, window_size])
        _, _, t_s, x_s, y_s  = img.shape
        t_ws, x_ws, y_ws = window_size

        
        assert t_s % (t_ws * 16) == 0, f'The size along the t-axis must be divisible by t_window_size * 16'
        assert x_s % (x_ws * 16) == 0, f'The size along the x-axis must be divisible by x_window_size * 16'
        assert y_s % (y_ws * 16) == 0, f'The size along the y-axis must be divisible by y_window_size * 16'

        down12_1 = self.enc12(img)              # (B, C, T//2, X//2, Y//2)
        down3 = self.enc3(down12_1)             # (B, 2C, T//4, X//4, Y//4)
        down4 = self.enc4(down3)                # (B, 4C, T//8, X//8, Y//8)
        features = self.enc5(down4)             # (B, 8C, T//16, X//16, Y//16)

        up4 = self.dec4(features)               # (B, 8C, T//8, X//8, Y//8)
        # Merge up4 and down4
        up4 = self.converge4(up4, down4)        # (B, 4C, T//8, X//8, Y//8)

        up3 = self.dec3(up4)                    # (B, 4C, T//4, X//4, Y//4)
        # Merge up3 and down3
        up3 = self.converge3(up3, down3)        # (B, 2C, T//4, X//4, Y//4)

        up12 = self.dec12(up3)                  # (B, 2C, T//2, X//2, Y//2)
        # Merge up12 and down12_1
        up12 = self.converge12(up12, down12_1)  # (B, C, T//2, X//2, Y//2)

        out = self.final(up12)                  # (B, num_classes, T, X, Y)
        out = self.out(out)
        return out

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv3d, nn.ConvTranspose3d)):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)