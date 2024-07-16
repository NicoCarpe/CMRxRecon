import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import glob
import os

def get_loss(loss):
    return eval(loss)

def get_metrics():
    return [loss_rmse, loss_abs_mse, loss_abs_mae]

def complex_abs(x):
    return torch.sqrt(x.real**2 + x.imag**2)

def loss_abs_mse(y_true, y_pred):
    y_true = y_true.to(torch.complex64)
    y_pred = y_pred.to(torch.complex64)
    diff = (complex_abs(y_true) - complex_abs(y_pred))
    return torch.mean(torch.sum(diff.real**2, dim=(1, 2, 3)), dim=(0, -1))

def loss_complex_mae_all_dim(y_true, y_pred):
    y_true = y_true.to(torch.complex64)
    y_pred = y_pred.to(torch.complex64)
    diff = (y_true - y_pred)
    return torch.mean(torch.sum(torch.sqrt(diff.real**2 + diff.imag**2 + 1e-9)))

def loss_abs_mae(y_true, y_pred):
    y_true = y_true.to(torch.complex64)
    y_pred = y_pred.to(torch.complex64)
    diff = (complex_abs(y_true) - complex_abs(y_pred))
    return torch.mean(torch.sum(torch.sqrt(diff.real**2 + diff.imag**2 + 1e-9), dim=(1, 2, 3)), dim=(0, -1))

def loss_rmse(y_true, y_pred):
    y_true = y_true.to(torch.complex64)
    y_pred = y_pred.to(torch.complex64)
    diff = (complex_abs(y_true).float() - complex_abs(y_pred).float())
    nominator = torch.sum(diff.real**2, dim=(1, 2, 3, 4))
    denominator = torch.sum(y_true.real**2 + y_true.imag**2, dim=(1, 2, 3, 4))
    return torch.mean(torch.sqrt(nominator / denominator))

def get_checkpoint(exp_dir, model_name):
    latest = max(glob.glob(f'{exp_dir}/{model_name}*/weights*.pt'), key=os.path.getmtime)
    print(f'Use latest checkpoint: {latest}')
    return latest

def get_optimizer_checkpoint(exp_dir, model_name):
    latest = max(glob.glob(f'{exp_dir}/{model_name}*/optimizer*.pkl'), key=os.path.getmtime)
    print(f'Use latest checkpoint: {latest}')
    return latest
