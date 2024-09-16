#!/usr/bin/env python3
"""
refer to the work of XuanKai: https://github.com/woxuankai/SpatialAlignmentNetwork
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn

def ckpt_load(folder):
    if os.path.isfile(folder):
        return torch.load(folder)
    ckpt = {}
    for key in os.listdir(folder):
        save_path = os.path.join(folder, key)
        if key == 'config':
            try:
                ckpt[key] = Config()
                ckpt[key].load(save_path)
            except UnicodeDecodeError:
                ckpt[key] = torch.load(save_path)
        else:
            try:
                ckpt[key] = torch.load(save_path, map_location='cpu')
            except RuntimeError as e:
                ckpt[key] = np.load(save_path)
                ckpt[key] = {k: torch.from_numpy(v) for k, v in ckpt[key].items()}
    return ckpt

def ckpt_save(ckpt, folder):
    assert isinstance(ckpt, dict)
    if not os.path.exists(folder):
        os.mkdir(folder)
    for key, val in ckpt.items():
        save_path = os.path.join(folder, key)
        if key == 'config':
            val.save(save_path)
        else:
            val = {k: v.cpu().numpy() for k, v in val.items()}
            with open(save_path, 'wb') as f:
                np.savez(f, **val)

class Config(object):
    def __init__(self, **params):
        super().__init__()
        super().__setattr__('memo', [])
        for key, val in params.items():
            setattr(self, key, val)

    def __setattr__(self, name, value):
        if name not in self.memo:
            self.memo.append(name)
        super().__setattr__(name, value)

    def __delattr__(self, name):
        self.memo.remove(name)
        super().__delattr__(name)

    def __str__(self):
        return 'class Config containing: ' + str({key: getattr(self, key) for key in self.memo})

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, param):
        assert param in self.memo, str(param) + ' not found, try ' + str(self.memo)
        return getattr(self, param)

    def __contains__(self, item):
        return item in self.memo

    def load(self, save_path):
        for k in self.memo.copy():
            self.pop(k)
        with open(save_path, 'r') as f:
            content = json.load(f)
        for k, v in content.items():
            setattr(self, k, v)

    def save(self, save_path):
        content = {k: getattr(self, k) for k in self.memo}
        with open(save_path, 'w') as f:
            json.dump(content, f)

class BaseModel(nn.Module):
    def __init__(self, ckpt=None, objects=None, **kwargs):
        super(BaseModel, self).__init__()
        self.kwargs = kwargs  # Store kwargs instead of cfg

        if ckpt is not None:
            self.load(ckpt=ckpt, objects=objects)
        else:
            self.build()

        self.training = True

    def build(self):
        # Initialize model using kwargs
        for key, value in self.kwargs.items():
            setattr(self, key, value)

    def to(self, device):
        for value in self.__dict__.values():
            if isinstance(value, torch.nn.Module) or isinstance(value, torch.Tensor):
                value.to(device)
            if isinstance(value, torch.optim.Optimizer):
                for param in value.state.values():
                    if isinstance(param, torch.Tensor):
                        param.data = param.data.to(device)
                        if param._grad is not None:
                            param._grad.data = param._grad.data.to(device)
                    elif isinstance(param, dict):
                        for subparam in param.values():
                            if isinstance(subparam, torch.Tensor):
                                subparam.data = subparam.data.to(device)
                                if subparam._grad is not None:
                                    subparam._grad.data = subparam._grad.data.to(device)
        return self

    def train(self, mode=True):
        for value in self.__dict__.values():
            if isinstance(value, torch.nn.Module):
                value.train(mode)
        self.training = mode
        return self

    def eval(self):
        for value in self.__dict__.values():
            if isinstance(value, torch.nn.Module):
                value.eval()
        self.training = False
        return self

    def get_saveable(self):
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            return {key: value for key, value in modules.items() if isinstance(value, torch.nn.Module)}
        else:
            return {key: value for key, value in self.__dict__.items() if isinstance(value, torch.nn.Module)}

    def save(self, ckpt, objects=None):
        saveable = self.get_saveable()
        if objects is None:
            objects = saveable.keys()
        objects = {key: saveable[key].state_dict() for key in objects}
        objects['kwargs'] = self.kwargs  # Save kwargs for model reconstruction
        ckpt_save(objects, ckpt)

    def load(self, ckpt, objects=None):
        ckpt = ckpt_load(ckpt)
        self.kwargs = ckpt.pop('kwargs', {})  # Load kwargs from checkpoint
        self.build()  # Rebuild model using loaded kwargs

        saveable = self.get_saveable()
        if objects is None:
            objects = saveable.keys()
        objects = {key: saveable[key] for key in objects}
        for key, value in objects.items():
            value.load_state_dict(ckpt[key])

if __name__ == '__main__':
    import sys
    import shutil

    ckpt = ckpt_load(sys.argv[1])
    if len(sys.argv) >= 3:
        ckpt_save(ckpt, sys.argv[2])
    else:
        if os.path.isdir(sys.argv[1]):
            shutil.rmtree(sys.argv[1])
        elif os.path.isfile(sys.argv[1]):
            os.remove(sys.argv[1])
        else:
            assert False
        ckpt_save(ckpt, sys.argv[1])