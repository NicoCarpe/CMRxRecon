import numpy as np
import skimage
try:
    import skimage.metrics
    from skimage.metrics import structural_similarity as compare_ssim
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
except ImportError:
    import skimage.measure
    from skimage.measure import compare_psnr, compare_ssim


def to_numpy(*args):
    outputs = []
    for arg in args:
        if hasattr(arg, 'cpu') and callable(arg.cpu):
            arg = arg.detach().cpu()
        if hasattr(arg, 'numpy') and callable(arg.numpy):
            arg = arg.detach().numpy()
        assert len(arg.shape) == 5, 'wrong shape [batch, temporal, channel=1, rows, cols]'
        outputs.append(arg)
    return outputs


def mse(gt, pred, gt_flag=True):
    if gt_flag:
        gt, pred = to_numpy(gt, pred)
        mse = np.mean((gt - pred) ** 2).item()
    else:
        mse = -1000
    return mse


def mae(gt, pred, gt_flag=True):
    if gt_flag:
        gt, pred = to_numpy(gt, pred)
        mae = np.mean(np.absolute(gt - pred)).item()
    else:
        mae = -1000
    return mae


def nmse(gt, pred, gt_flag=True):
    if gt_flag:
        gt, pred = to_numpy(gt, pred)
        nmse = (np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2).item()
    else:
        nmse = -1000
    return nmse


def psnr(gt, pred, gt_flag=True):
    if gt_flag:
        gt, pred = to_numpy(gt, pred)
        psnr = np.mean([compare_psnr(g[0], p[0], data_range=1) for g, p in zip(gt, pred)]).item()
    else:
        psnr = -1000
    return psnr


def ssim(gt, pred, gt_flag=True):
    if gt_flag:
        gt, pred = to_numpy(gt, pred)
        ssim = np.mean([compare_ssim(g[0], p[0], data_range=1) for g, p in zip(gt, pred)]).item()
    else:
        ssim = -1000
    return ssim


if __name__ == "__main__":
    gt, pred = np.random.rand(10, 5, 1, 100, 100), np.random.rand(10, 5, 1, 100, 100)
    print('MSE', mse(gt, pred))
    print('NMSE', nmse(gt, pred))
    print('PSNR', psnr(gt, pred))
    print('SSIM', ssim(gt, pred))
